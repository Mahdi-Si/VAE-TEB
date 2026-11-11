r"""Pilot fine-tune harness: one fixed budget, one data order per seed, one scoring pass.

A pilot answers one question: does a change to the model or to the objective help when training
continues from the epoch-424 weights? Every arm therefore gets the same start, the same segments
in the same order, the same optimizer and the same number of steps. Only the arm's own patch
differs. A control arm with no patch absorbs the effect of fine-tuning itself, so an arm is
always read against the control of the same seed.

**Data.** The fit recordings (split code 0) of the three local shards are loaded once into
memory. A seeded generator draws the segment order, so two arms with one seed see the same
batches. The scored recordings (split code 2 for the pilots) are never trained on.

**Step.** The task's own ``compute_loss_and_metrics(batch, 0, 'train')`` builds the loss, so the
objective is the training objective: the tiled anchor set with the hashed phase, the event
weight, the source-null penalty and source dropout. The KL weight is the run's value after its
warm-up, $\beta = 1$; the harness presents a training epoch above the warm-up length to the task
and advances it once per pass over the fit set, which also rotates the anchor phase as in
training.

**Scoring.** :func:`evaluate` decodes both branches at their latent means on anchors every 5
steps and returns one row per anchor:

* $\mathrm{NLL}_{\text{base}}$, $\mathrm{NLL}_{\text{full}}$ and the source gain
  $G = \mathrm{NLL}_{\text{base}} - \mathrm{NLL}_{\text{full}}$, in nats per anchor block;
* the KL, the gate, and the source-null KL (UA values zeroed, flags kept);
* sums of squared errors per block, on the same cells as the linear baselines;
* standardised AR(1) innovations summarised as counts inside 1, 2 and 3 standard deviations per
  block, for the calibration check.
"""
from __future__ import annotations

import copy
import math
import types
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from teb_vae.lag_attn_cfs.eval.config_schema import force_single_process_loader, merge_eval_overrides
from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.nets.losses import ar_innovation
from teb_vae.neural_surgery_trf_cfs.session7 import s7_interventions as I
from teb_vae.neural_surgery_trf_cfs.session7 import s7common as S
from train.data_module import GraphDataModule

#: The fixed pilot budget. Every arm of one comparison uses these values.
BUDGET: Dict[str, Any] = {
    "steps": 300,            # optimizer steps
    "batch_size": 64,        # segments per forward
    "accumulate": 2,         # forwards per optimizer step
    "lr": 1e-4,
    "weight_decay": None,    # None: the task's own value
    "grad_clip": 100.0,      # global norm clip; generous, it only stops a rare spike
    "epoch_offset": 424,     # the training epoch presented to the task at pass 0
}

N_ST = 32


class SegmentStore:
    """The segments of a GUID set, loaded once and served as collated batches.

    Attributes:
        items: One loader item per segment.
        collate: The data module's collate function.
        guids: The GUID of each item.
    """

    def __init__(self, cfg: dict, guids: Sequence[str]) -> None:
        """Load every segment of ``guids`` from the three local shards.

        Args:
            cfg: The merged evaluation configuration.
            guids: The recordings to keep.
        """
        cfg = copy.deepcopy(cfg)
        cfg["dataset_config"]["dataloader_config"]["dataset_kwargs"]["allowed_guids"] = list(guids)
        loader = GraphDataModule(cfg).test_dataloader()
        self.collate = loader.collate_fn
        dataset = loader.dataset
        self.items = [dataset[index] for index in range(len(dataset))]
        self.guids = [S.guid_text(self._guid(item)) for item in self.items]

    @staticmethod
    def _guid(item: Any) -> Any:
        """Read the GUID of one loader item, whatever container the dataset returns."""
        return item["guid"] if isinstance(item, dict) or hasattr(item, "keys") else getattr(item, "guid")

    def __len__(self) -> int:
        return len(self.items)

    def batch(self, index: Sequence[int]) -> Any:
        """Collate the items at ``index`` into one batch.

        Args:
            index: Item positions.

        Returns:
            The collated batch, on the host.
        """
        return self.collate([self.items[i] for i in index])

    def ordered(self, batch_size: int):
        """Yield every segment once, in storage order.

        Args:
            batch_size: Segments per batch.

        Yields:
            Collated batches.
        """
        for start in range(0, len(self.items), batch_size):
            yield self.batch(range(start, min(start + batch_size, len(self.items))))


def guids_with_code(codes: Sequence[int], cohorts: Optional[Sequence[str]] = None) -> List[str]:
    """List the local recordings with the given split codes.

    Args:
        codes: Split codes to keep.
        cohorts: Cohort names to keep; ``None`` keeps all three.

    Returns:
        Sorted GUIDs.
    """
    table = S.cohort_of_guids()
    return sorted(g for g, c in table.items()
                  if S.guid_hash(g) in codes and (cohorts is None or c in cohorts))


def load_eval_config() -> dict:
    """The merged evaluation configuration with a single-process loader.

    Returns:
        The configuration dictionary.
    """
    cfg = merge_eval_overrides(load_config(str(S.CONFIG)), str(S.OVERRIDES))
    force_single_process_loader(cfg)
    return cfg


def present_epoch(task: Any, epoch: int) -> None:
    """Make the task read ``epoch`` as the current training epoch.

    The task reads the epoch for the KL warm-up and for the hashed anchor phase. Outside a
    Lightning trainer it would read 0, which is the first warm-up epoch with $\beta = 0$.

    Args:
        task: The Lightning task.
        epoch: The epoch to present.
    """
    task._trainer = types.SimpleNamespace(current_epoch=int(epoch))


def fine_tune(task: Any, store: SegmentStore, dev: torch.device, seed: int,
              budget: Dict[str, Any] = BUDGET,
              on_step: Optional[Callable[[int, Dict[str, float]], None]] = None) -> List[Dict[str, float]]:
    """Fine-tune the task's model in place for the fixed budget.

    Args:
        task: The task, already patched by the arm.
        store: The fit segments.
        dev: Training device.
        seed: Seed of the data order, of dropout and of the latent noise.
        budget: The step budget; see :data:`BUDGET`.
        on_step: Optional callback ``(step, metrics)`` after each optimizer step.

    Returns:
        One metrics dictionary per optimizer step (loss, KL, gate and reconstruction terms).
    """
    model = task.orig_model
    torch.manual_seed(seed)
    np.random.seed(seed)
    order_rng = np.random.default_rng(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    weight_decay = budget["weight_decay"]
    if weight_decay is None:
        weight_decay = float(getattr(task.hparams, "weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(params, lr=float(budget["lr"]), weight_decay=weight_decay, eps=1e-8)
    history: List[Dict[str, float]] = []
    per_pass = max(1, len(store) // int(budget["batch_size"]))
    forwards = int(budget["steps"]) * int(budget["accumulate"])
    order: List[int] = []
    model.train()
    task.train()
    optimizer.zero_grad(set_to_none=True)
    for forward in range(forwards):
        if forward % per_pass == 0:
            order = order_rng.permutation(len(store)).tolist()
            present_epoch(task, int(budget["epoch_offset"]) + forward // per_pass)
        start = (forward % per_pass) * int(budget["batch_size"])
        batch = task.transfer_batch_to_device(store.batch(order[start:start + int(budget["batch_size"])]), dev, 0)
        loss, metrics = task.compute_loss_and_metrics(batch, 0, "train")
        (loss / int(budget["accumulate"])).backward()
        if (forward + 1) % int(budget["accumulate"]) == 0:
            torch.nn.utils.clip_grad_norm_(params, float(budget["grad_clip"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            row = {key: float(value) for key, value in metrics.items()
                   if isinstance(value, (float, int)) or (torch.is_tensor(value) and value.numel() == 1)}
            row["step"] = (forward + 1) // int(budget["accumulate"])
            history.append(row)
            if on_step is not None:
                on_step(row["step"], row)
    model.eval()
    task.eval()
    return history


@torch.no_grad()
def evaluate(task: Any, store: SegmentStore, dev: torch.device, anchor_stride: int = 5,
             batch_size: int = 32, nu: Optional[float] = None) -> pd.DataFrame:
    r"""Score the model on a segment store with mean decoding; one row per anchor.

    Args:
        task: The task whose model is scored.
        store: The scored segments.
        dev: Device.
        anchor_stride: Spacing between decoded anchors.
        batch_size: Segments per forward.
        nu: Degrees of freedom of a Student-$t$ innovation density; ``None`` scores the Gaussian
            density. The log score is a proper scoring rule, so the NLL of a Student-$t$ arm and
            of a Gaussian arm are comparable. The three coverage counts use the central
            intervals of the arm's own density at the Gaussian 1, 2 and 3 standard-deviation
            levels (68.27%, 95.45%, 99.73%).

    Returns:
        The per-anchor table described in the module docstring.
    """
    model = task.orig_model
    model.eval()
    lk = model.forecast_likelihood_kwargs()
    cell_mask = lk["cell_mask"]
    frames: List[pd.DataFrame] = []
    for raw in store.ordered(batch_size):
        batch = task.transfer_batch_to_device(raw, dev, 0)
        m = I.matched_forward(task, model, batch, anchor_stride=anchor_stride)
        out, anchors = m.out, m.anchors
        valid = m.anchor_valid.cpu().numpy().reshape(-1)
        columns: Dict[str, np.ndarray] = {}

        def put(name: str, tensor: torch.Tensor) -> None:
            columns[name] = tensor.detach().float().cpu().numpy().reshape(-1)[valid]

        counted = (cell_mask[None, None] * m.step_mask[..., None])                 # (B, A, H, C)
        for branch, latent in (("base", out["mu_prior"]), ("full", out["mu_post"])):
            z = I.at_anchors(latent, anchors)
            mu, logvar = model.decoder(z, persistence=out.get("persistence"))
            residual = m.target - mu
            innovation, innovation_logvar = ar_innovation(residual, lk["ar_coef"], m.step_mask, logvar)
            z2 = innovation.pow(2) * torch.exp(-innovation_logvar)
            nll = innovation_nll(z2, innovation_logvar, nu) * counted
            put(f"nll_{branch}", nll.sum(dim=(-1, -2)))
            put(f"nll_{branch}_st", nll[..., :N_ST].sum(dim=(-1, -2)))
            put(f"nll_{branch}_ph", nll[..., N_ST:].sum(dim=(-1, -2)))
            squared = residual.pow(2) * counted
            for block, cols in (("st", slice(0, N_ST)), ("ph", slice(N_ST, None))):
                put(f"sse_{block}_{branch}", squared[..., cols].sum(dim=(-1, -2)))
                put(f"sse0_{block}_{branch}", squared[:, :, 0, cols].sum(dim=-1))
                put(f"sse5_{block}_{branch}", squared[:, :, :5, cols].sum(dim=(-1, -2)))
                if branch == "full":
                    put(f"n_{block}", counted[..., cols].sum(dim=(-1, -2)))
                    put(f"n0_{block}", counted[:, :, 0, cols].sum(dim=-1))
                    put(f"n5_{block}", counted[:, :, :5, cols].sum(dim=(-1, -2)))
                    for k, bound in zip((1, 2, 3), interval_bounds(nu)):
                        put(f"in{k}_{block}", ((z2 <= bound * bound).float() * counted)[..., cols].sum(dim=(-1, -2)))
                    floor = float(model.decoder.logvar_clamp[0]) if hasattr(model.decoder, "logvar_clamp") else -6.0
                    put(f"floor_{block}", ((logvar <= floor + 1e-3).float() * counted)[..., cols].sum(dim=(-1, -2)))
        kld = model.kld_tensor(mu_prior=out["mu_prior"], logvar_prior=out["logvar_prior"],
                               mu_post=out["mu_post"], logvar_post=out["logvar_post"])
        put("kl", I.at_anchors(kld, anchors).sum(dim=-1))
        put("g", I.at_anchors(out["event_gate"], anchors)[..., 0])
        null = I.reattend(model, m, source=torch.zeros_like(m.source))
        put("kl_null", I.anchor_kl(model, m, null["mu_post"], null["logvar_post"]))
        put("keys_invalid", I.at_anchors(m.lag_mask.to(torch.bool)[None] & ~m.key_mask, anchors).float().sum(dim=-1))
        count = anchors.shape[1]
        guids = [S.guid_text(g) for g in batch["guid"]]
        columns["guid"] = np.repeat(np.asarray(guids, dtype=object), count)[valid]
        columns["epoch"] = np.repeat(batch["epoch"].float().cpu().numpy(), count)[valid]
        columns["anchor"] = anchors.cpu().numpy().reshape(-1)[valid]
        frames.append(pd.DataFrame(columns))
    table = pd.concat(frames, ignore_index=True)
    table["gain"] = table["nll_base"] - table["nll_full"]
    return table


def innovation_nll(z2: torch.Tensor, logvar: torch.Tensor, nu: Optional[float] = None) -> torch.Tensor:
    r"""Negative log-density of a standardised innovation, Gaussian or Student-$t$.

    With $z^2 = \varepsilon^2 / \sigma^2$ and $\log\sigma^2$ the (carried) log-variance, the Gaussian
    term is $\tfrac12(\log 2\pi + \log\sigma^2 + z^2)$. The Student-$t$ term with scale $\sigma$ is

    $$-\log\Gamma\!\left(\tfrac{\nu+1}{2}\right) + \log\Gamma\!\left(\tfrac{\nu}{2}\right)
      + \tfrac12\log(\nu\pi) + \tfrac12\log\sigma^2 + \tfrac{\nu+1}{2}\log\!\left(1 + \tfrac{z^2}{\nu}\right).$$

    Args:
        z2: Squared standardised innovation.
        logvar: Log of the squared scale.
        nu: Degrees of freedom; ``None`` is the Gaussian density.

    Returns:
        A tensor shaped like ``z2``.
    """
    if nu is None:
        return 0.5 * (math.log(2 * math.pi) + logvar + z2)
    nu = float(nu)
    constant = -math.lgamma((nu + 1) / 2) + math.lgamma(nu / 2) + 0.5 * math.log(nu * math.pi)
    return constant + 0.5 * logvar + 0.5 * (nu + 1) * torch.log1p(z2 / nu)


def interval_bounds(nu: Optional[float] = None) -> Sequence[float]:
    r"""Half-widths, in scale units, of the central 68.27%, 95.45% and 99.73% intervals.

    Args:
        nu: Degrees of freedom of the Student-$t$ density; ``None`` is the Gaussian density.

    Returns:
        Three half-widths: $(1, 2, 3)$ for the Gaussian density, the matching quantiles otherwise.
    """
    if nu is None:
        return (1.0, 2.0, 3.0)
    from scipy import stats

    return tuple(float(stats.t.ppf(0.5 + level / 2, float(nu))) for level in (0.6826895, 0.9544997, 0.9973002))


def calibration(table: pd.DataFrame) -> Dict[str, float]:
    """Coverage of the 1, 2 and 3 standard-deviation intervals of the full-branch innovations.

    Args:
        table: A table from :func:`evaluate`.

    Returns:
        Coverage per block and level, and the largest absolute gap to the Gaussian nominal.
    """
    nominal = {1: 0.6827, 2: 0.9545, 3: 0.9973}
    result: Dict[str, float] = {}
    worst = 0.0
    for block in ("st", "ph"):
        cells = table[f"n_{block}"].sum()
        for k, target in nominal.items():
            coverage = float(table[f"in{k}_{block}"].sum() / max(cells, 1.0))
            result[f"cov{k}_{block}"] = coverage
            worst = max(worst, abs(coverage - target))
        result[f"floor_{block}"] = float(table[f"floor_{block}"].sum() / max(cells, 1.0))
    result["max_abs_gap"] = worst
    # The official verdict's statistic: the largest relative error of the tail mass, pooled over
    # both blocks, against a tolerance of 0.5.
    cells = table["n_st"].sum() + table["n_ph"].sum()
    relative = 0.0
    for k, target in nominal.items():
        tail = 1.0 - float((table[f"in{k}_st"].sum() + table[f"in{k}_ph"].sum()) / max(cells, 1.0))
        result[f"tail{k}"] = tail
        relative = max(relative, abs(tail - (1.0 - target)) / (1.0 - target))
    result["worst_relative_tail_error"] = relative
    return result
