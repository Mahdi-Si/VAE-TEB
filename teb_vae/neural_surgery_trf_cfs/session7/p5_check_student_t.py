r"""Phase 5: check the production Student-$t$ forecast density against the session-7 pilot.

The production key is the model keyword ``forecast_student_nu`` (``None`` is the Gaussian
density). The script runs on the CPU, on 8 rows of the first evaluation batch: the first 8 for
the bitwise reference, and also the 8 with the most invalid UA samples for check b, so that step
masks with gaps inside a horizon window are exercised.

a. **Inert.** Run once with ``mode='reference'`` before the production edit. With the key at its
   default, the forward outputs, the validation loss and the validation metrics must then be
   bitwise the stored reference, and the state dict must carry the same keys.
b. **Equivalence.** A model built from the checkpoint's ``model_kwargs`` plus
   ``forecast_student_nu`` $= \nu$ loads the state dict strictly. Its production per-anchor block
   score must equal the pilot formula (``s7_pilot.innovation_nll`` on the same AR(1) innovations,
   under the cell mask and the step mask) to a relative $10^{-4}$. Every block the objective
   itself scores in a validation step (captured by a wrapper on the scorer) must carry
   $\nu$ and match the pilot formula on the captured tensors to the same bound, which shows that
   the objective scores under the Student-$t$ density. One training step must compute the loss
   and back-propagate.
c. **Calibration.** Synthetic innovations $\varepsilon = \sigma\, T$, $T \sim t_\nu$, with the
   model's own scale $\sigma$ and scored cells, go through the evaluation's calibration routine.
   Under ``student_nu`` $= \nu$ the observed masses must lie within $0.01$ of the nominal
   $0.6827, 0.9545, 0.9973$ and the ``calibration_near_nominal`` verdict must pass; under the
   Gaussian reading the same data must fail that verdict at the 3-sigma level. The closed-form
   CRPS is also checked against a Monte-Carlo estimate of $\tfrac12 \mathbb E|X - X'|$.

Results go to ``out/phase5/check_student_t.json``. Edit :data:`RUN_ARGS` and run this file from
the IDE, or pass ``--mode``.
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from teb_vae.neural_surgery_trf_cfs.session7 import p5_check_production as P5  # noqa: E402
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S  # noqa: E402

#: Run-button inputs. ``mode`` is ``'reference'`` (write the reference outputs of the unchanged
#: code) or ``'check'`` (run checks a to c against them). ``draws`` is how many independent
#: copies of the synthetic innovations check c pools, to shrink the sampling noise of the
#: 3-sigma tail mass.
RUN_ARGS: Dict[str, Any] = {"mode": "check", "rows": 8, "seed": 0, "nu": 8.0, "draws": 8}
OUT = S.OUT / "phase5"
REFERENCE = OUT / "reference_student_t.pt"
#: Relative bound of check b.
RELATIVE_BOUND = 1e-4
#: Absolute bound on each coverage mass in check c.
COVERAGE_BOUND = 0.01


def check_inert(task: Any, batch: Any, seed: int) -> Dict[str, Any]:
    """Check a: the default model reproduces the reference outputs bitwise.

    Args:
        task: The default task.
        batch: The first rows of the batch.
        seed: The seed.

    Returns:
        The record.
    """
    ref = torch.load(REFERENCE)
    rec = P5.run_outputs(task, batch, seed)
    out_keys = sorted(set(ref["outputs"]) | set(rec["outputs"]))
    mismatched = [k for k in out_keys if k not in ref["outputs"] or k not in rec["outputs"]
                  or not torch.equal(ref["outputs"][k], rec["outputs"][k])]
    metric_keys = sorted(set(ref["metrics"]) | set(rec["metrics"]))
    metric_mismatch = [k for k in metric_keys if k not in ref["metrics"] or k not in rec["metrics"]
                       or not torch.equal(ref["metrics"][k], rec["metrics"][k])]
    loss_equal = bool(torch.equal(ref["loss"], rec["loss"]))
    model = task.orig_model
    terms = model.forecast_likelihood_kwargs()
    return {
        "outputs_compared": len(out_keys),
        "outputs_mismatched": mismatched,
        "metrics_compared": len(metric_keys),
        "metrics_mismatched": metric_mismatch,
        "loss_bitwise_equal": loss_equal,
        "loss": float(rec["loss"]),
        "forecast_student_nu": getattr(model, "forecast_student_nu", "absent"),
        "likelihood_kwargs_keys": sorted(terms),
        "state_dict_keys": len(model.state_dict()),
        "pass": (not mismatched and not metric_mismatch and loss_equal
                 and sorted(terms) == ["ar_coef", "cell_mask"]),
    }


def pilot_block(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor,
                step_mask: torch.Tensor, ar_coef: torch.Tensor, cell_mask: torch.Tensor,
                nu: float, channel_weight: Optional[torch.Tensor] = None,
                horizon_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""The per-anchor block under the pilot's formula.

    The two objective weights are applied in the order of ``p4_pilots.patch_student_t``: the
    channel weight, then the horizon weight, then the cell mask.

    Args:
        mu: Forecast mean $(B, A, H, C)$.
        logvar: Forecast log-variance, the same shape.
        target: The gathered target $(B, A, H, C)$.
        step_mask: The forecast mask $(B, A, H)$.
        ar_coef: $\phi_c$.
        cell_mask: The $(H, C)$ scored-cell mask.
        nu: $\nu$.
        channel_weight: The objective's $(C,)$ channel weight, or ``None``.
        horizon_weight: The objective's $(H,)$ horizon weight, or ``None``.

    Returns:
        The block $(B, A)$.
    """
    from teb_vae.lag_attn_rws.nets.losses import ar_innovation
    from teb_vae.neural_surgery_trf_cfs.session7 import s7_pilot

    innovation, logvar_eff = ar_innovation(target - mu, ar_coef, step_mask, logvar)
    z2 = innovation ** 2 * torch.exp(-logvar_eff)
    per_cell = s7_pilot.innovation_nll(z2, logvar_eff, nu)
    if channel_weight is not None:
        per_cell = per_cell * channel_weight
    if horizon_weight is not None:
        per_cell = per_cell * horizon_weight[:, None]
    return (per_cell * cell_mask * step_mask[..., None]).sum(dim=(2, 3))


def max_relative(left: torch.Tensor, right: torch.Tensor, scored: torch.Tensor) -> float:
    """Largest relative difference over the scored anchors, against $\\max(|right|, 1)$.

    Args:
        left: First block $(B, A)$.
        right: Reference block $(B, A)$.
        scored: Boolean $(B, A)$ of the anchors to compare.

    Returns:
        The largest relative difference, ``nan`` when nothing is scored.
    """
    rel = ((left - right).abs() / right.abs().clamp_min(1.0))[scored]
    return float(rel.max()) if rel.numel() else float("nan")


def spy_objective_blocks(task: Any, batch: Any, seed: int) -> list:
    """Capture every block the objective scores during one validation step.

    The objective reaches the scorer through the losses module's global name, so a wrapper on
    that name sees the exact tensors, mask and keywords the objective uses. That avoids a second
    reconstruction of the objective's mask.

    Args:
        task: The task.
        batch: The batch.
        seed: The seed set before the step.

    Returns:
        One ``(args, kwargs, block, contributing)`` per call, every tensor detached.
    """
    from teb_vae.lag_attn_rws.nets import losses

    original = losses.masked_raw_block_per_anchor
    calls: list = []

    def spy(mu, target, mask, **kwargs):
        block, contributing = original(mu, target, mask, **kwargs)
        calls.append(((mu.detach(), target.detach(), mask.detach()),
                      {k: v.detach() if torch.is_tensor(v) else v for k, v in kwargs.items()},
                      block.detach(), contributing.detach()))
        return block, contributing

    losses.masked_raw_block_per_anchor = spy
    try:
        P5.present_epoch(task)
        with torch.no_grad():
            torch.manual_seed(seed)
            task.compute_loss_and_metrics(batch, 0, "val")
    finally:
        losses.masked_raw_block_per_anchor = original
    return calls


def check_equivalence(task: Any, batch: Any, seed: int, nu: float) -> Dict[str, Any]:
    """Check b (i): the production block equals the pilot's, and the objective scores it.

    Two comparisons. First, the production scorer under the model's own
    ``forecast_likelihood_kwargs`` on one forward against the pilot formula on the same tensors.
    Second, every block the objective itself scores in a validation step (both branches, and the
    shuffled control) against the pilot formula on the tensors the objective passed.

    Args:
        task: The task built with ``forecast_student_nu`` $= \\nu$.
        batch: Eight rows.
        seed: The seed set before the forward and before the loss.
        nu: $\\nu$.

    Returns:
        The record.
    """
    from teb_vae.lag_attn_rws.nets.losses import masked_raw_block_per_anchor
    from teb_vae.neural_surgery_trf_cfs import common

    model = task.orig_model
    P5.present_epoch(task)
    with torch.no_grad():
        torch.manual_seed(seed)
        _inputs, outputs, target, step_mask, _anchors, _valid = common.forward_parts(task, model, batch)
        terms = model.forecast_likelihood_kwargs()
        production, contributing = masked_raw_block_per_anchor(
            outputs["mu_full"], target, step_mask, likelihood="gaussian_nll",
            logvar=outputs["logvar_full"], **terms,
        )
        gaussian, _ = masked_raw_block_per_anchor(
            outputs["mu_full"], target, step_mask, likelihood="gaussian_nll",
            logvar=outputs["logvar_full"], cell_mask=terms["cell_mask"], ar_coef=terms["ar_coef"],
        )
        pilot = pilot_block(outputs["mu_full"], outputs["logvar_full"], target, step_mask,
                            terms["ar_coef"], terms["cell_mask"], nu)
    scored = contributing > 0
    production_error = max_relative(production, pilot, scored)

    objective_errors = []
    objective_nu = []
    for (mu, obj_target, mask), kwargs, block, obj_contributing in spy_objective_blocks(task, batch, seed):
        objective_nu.append(kwargs.get("student_nu"))
        reference = pilot_block(mu, kwargs["logvar"], obj_target, mask, kwargs["ar_coef"],
                                kwargs["cell_mask"], nu, kwargs.get("channel_weight"),
                                kwargs.get("horizon_weight"))
        objective_errors.append(max_relative(block, reference, obj_contributing > 0))
    # Masked steps between two valid steps of one window: these exercise the marginalisation.
    valid = step_mask > 0.5
    first = torch.where(valid.any(-1), valid.float().argmax(-1), torch.zeros_like(valid[..., 0], dtype=torch.long))
    last = valid.shape[-1] - 1 - torch.flip(valid, dims=[-1]).float().argmax(-1)
    steps = torch.arange(valid.shape[-1])
    inside = (steps >= first[..., None]) & (steps <= last[..., None])
    gaps = int((inside & ~valid & valid.any(-1, keepdim=True)).sum())
    return {
        "student_nu": terms.get("student_nu"),
        "likelihood_kwargs_keys": sorted(terms),
        "scored_anchors": int(scored.sum()),
        "interior_gap_steps": gaps,
        "max_relative_error_vs_pilot": production_error,
        "mean_block_student_t": float(production[scored].mean()),
        "mean_block_gaussian": float(gaussian[scored].mean()),
        "objective_blocks_scored": len(objective_errors),
        "objective_student_nu_seen": objective_nu,
        "objective_max_relative_error_vs_pilot": max(objective_errors) if objective_errors else float("nan"),
        "pass": bool(production_error <= RELATIVE_BOUND
                     and objective_errors and max(objective_errors) <= RELATIVE_BOUND
                     and all(value == nu for value in objective_nu)
                     and not torch.allclose(production, gaussian)),
    }


def check_training(task: Any, batch: Any, seed: int) -> Dict[str, Any]:
    """Check b (ii): one training step under the Student-$t$ density back-propagates.

    Args:
        task: The task built with ``forecast_student_nu``.
        batch: Eight rows.
        seed: The seed.

    Returns:
        The record.
    """
    task._trainer = types.SimpleNamespace(current_epoch=424)
    task.train()
    task.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    loss, metrics = task.compute_loss_and_metrics(batch, 0, "train")
    loss.backward()
    grads = [p.grad for p in task.parameters() if p.grad is not None]
    finite = all(bool(torch.isfinite(g).all()) for g in grads)
    nonzero = sum(int(bool(g.abs().sum() > 0)) for g in grads)
    task.eval()
    return {
        "train_loss": float(loss.detach()),
        "nll_full_block": float(metrics["nll_full_block"]),
        "params_with_grad": len(grads),
        "params_with_nonzero_grad": nonzero,
        "grads_finite": finite,
        "pass": bool(torch.isfinite(loss.detach()) and finite and nonzero > 0),
    }


def check_calibration(task: Any, batch: Any, seed: int, nu: float, draws: int) -> Dict[str, Any]:
    r"""Check c: Student-$t$ innovations read calibrated under $\nu$ and not under the Gaussian.

    Args:
        task: The task built with ``forecast_student_nu`` $= \nu$ (for the scale and the cells).
        batch: Eight rows.
        seed: The seed of the synthetic draw.
        nu: $\nu$.
        draws: Independent copies pooled.

    Returns:
        The record.
    """
    from teb_vae.lag_attn_cfs.eval import metrics as M
    from teb_vae.neural_surgery_trf_cfs import common

    model = task.orig_model
    with torch.no_grad():
        torch.manual_seed(seed)
        _inputs, outputs, target, step_mask, _anchors, _valid = common.forward_parts(task, model, batch)
    logvar = outputs["logvar_full"].detach()
    cell_mask = model.forecast_likelihood_kwargs()["cell_mask"]
    clamp = M.decoder_logvar_bound(model)
    readings: Dict[str, Dict[str, torch.Tensor]] = {"student_t": {}, "gaussian": {}}
    for copy in range(int(draws)):
        # StudentT.sample reads the global stream, so each copy seeds it on its own.
        torch.manual_seed(seed + 1 + copy)
        t_draw = torch.distributions.StudentT(torch.tensor(float(nu))).sample(logvar.shape)
        innovation = torch.exp(0.5 * logvar) * t_draw
        # mu = 0, target = innovation and no AR term: the routine then reads these innovations
        # directly, under the model's own scale and scored cells.
        zero = torch.zeros_like(innovation)
        for name, density in (("student_t", nu), ("gaussian", None)):
            sums = M.calibration_sums(zero, logvar, innovation, step_mask, logvar_clamp=clamp,
                                      cell_mask=cell_mask, ar_coef=None, student_nu=density)
            for key, value in sums.items():
                readings[name][key] = value if key not in readings[name] else readings[name][key] + value
    record: Dict[str, Any] = {"nu": nu, "draws": int(draws),
                              "half_widths_student_t": list(M.coverage_half_widths(nu))}
    for name, density in (("student_t", nu), ("gaussian", None)):
        report = M.calibration_report(readings[name], logvar_clamp=clamp, student_nu=density)
        verdict = M._calibration_verdict(report, tail_tolerance=M.DEFAULT_COVERAGE_TAIL_TOLERANCE)
        record[name] = {
            "n_coefficients": report["n_coefficients"],
            "coverage": [{k: r[k] for k in ("level_sigma", "nominal", "half_width", "observed")}
                         for r in report["coverage"]],
            "pit_max_cdf_deviation": report["pit"]["max_cdf_deviation"],
            "mean_standardised_sq": report["mean_standardised_sq"],
            "crps_normalised": report["crps_normalised"],
            "model_nll_per_coefficient": report["nll"]["model_per_coefficient"],
            "verdict": verdict.status if hasattr(verdict, "status") else str(verdict),
            "verdict_values": getattr(verdict, "values", None),
        }
    # The closed-form CRPS against Monte Carlo: for X ~ F, E CRPS(F, X) = E|X - X'| / 2.
    dist = torch.distributions.StudentT(torch.tensor(float(nu), dtype=torch.float64))
    torch.manual_seed(seed + 1000)
    x = dist.sample((2_000_000,))
    x_prime = dist.sample((2_000_000,))
    closed = float(M.student_t_crps(x, nu).mean())
    monte_carlo = 0.5 * float((x - x_prime).abs().mean())
    record["crps_closed_form_mean"] = closed
    record["crps_monte_carlo_half_mean_abs"] = monte_carlo
    student = record["student_t"]
    gaussian = record["gaussian"]
    record["pass"] = bool(
        all(abs(r["observed"] - r["nominal"]) <= COVERAGE_BOUND for r in student["coverage"])
        and student["verdict"] == "PASS"
        and gaussian["verdict"] == "FAIL"
        and gaussian["verdict_values"].get("worst_level_sigma") == 3.0
        and abs(closed - monte_carlo) <= 2e-3
    )
    return record


def main(mode: str, rows: int, seed: int, nu: float, draws: int) -> Dict[str, Any]:
    """Write the reference outputs, or run checks a to c.

    Args:
        mode: ``'reference'`` or ``'check'``.
        rows: Rows of the batch.
        seed: The seed.
        nu: Degrees of freedom $\\nu$ of the checked model.
        draws: Copies of the synthetic innovations in check c.

    Returns:
        The record.
    """
    dev, cfg, task = S.load_model(device="cpu")
    full = P5.first_batch(task, cfg, dev)
    batch = P5.select(full, range(rows))
    OUT.mkdir(parents=True, exist_ok=True)
    if mode == "reference":
        record = P5.run_outputs(task, batch, seed)
        torch.save(record, REFERENCE)
        print("saved", REFERENCE, "loss", float(record["loss"]))
        return {"reference": str(REFERENCE)}

    hard = P5.select(full, P5.most_invalid_rows(full, rows))
    task_t = P5.build_task(model_extra={"forecast_student_nu": float(nu)})
    default_keys = sorted(task.orig_model.state_dict())
    report: Dict[str, Any] = {
        "rows": rows, "seed": seed, "nu": nu,
        "state_dict_keys_equal": sorted(task_t.orig_model.state_dict()) == default_keys,
    }
    for name, run in (("a_inert", lambda: check_inert(task, batch, seed)),
                      ("b_equivalence_first_rows", lambda: check_equivalence(task_t, batch, seed, nu)),
                      ("b_equivalence_hard_rows", lambda: check_equivalence(task_t, hard, seed, nu)),
                      ("b_training", lambda: check_training(task_t, hard, seed)),
                      ("c_calibration", lambda: check_calibration(task_t, batch, seed, nu, draws))):
        report[name] = run()
        print(name, json.dumps(report[name], indent=1, default=str))
    (OUT / "check_student_t.json").write_text(json.dumps(report, indent=1, default=str))
    passed = report["state_dict_keys_equal"] and all(
        report[k]["pass"] for k in report if k[:2] in ("a_", "b_", "c_"))
    print("ALL PASS" if passed else "SOME CHECK FAILED")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=RUN_ARGS["mode"], choices=("reference", "check"))
    parser.add_argument("--rows", type=int, default=RUN_ARGS["rows"])
    parser.add_argument("--seed", type=int, default=RUN_ARGS["seed"])
    parser.add_argument("--nu", type=float, default=RUN_ARGS["nu"])
    parser.add_argument("--draws", type=int, default=RUN_ARGS["draws"])
    args = parser.parse_args()
    main(args.mode, args.rows, args.seed, args.nu, args.draws)
