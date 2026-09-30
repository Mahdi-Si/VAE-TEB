r"""Feature sources, the frozen feature cache and the per-channel scaler (SPEC §8).

Two sources turn one collated ``CombinedHDF5Dataset`` batch into :class:`StepFeatures` on the 4-s
grid: :class:`VaeSource`, a frozen VAE's per-step outputs (§8.2), and :class:`Hdf5Source`, the
stored ST/PH/raw streams (§8.3). There is no base class; both carry ``step_seconds``,
``required_fields``, ``trainable``, ``fingerprint``, ``dataset(paths)`` and ``__call__(batch)``.
Invalid steps are zeroed and ``time_pool`` is applied inside ``__call__``, so the cache and an
online consumer see the same tensors.

:func:`extract` runs a source once per unique ``(guid, epoch_s)`` of the cohort segment table into
``cache_root/<fingerprint hash>/`` (§8.4); :func:`open_cache` and :func:`read_rows` read it back.
:func:`fit_scaler` is the §8.5 train-only, recording-weighted scaler. Transforms (``log1p``,
``asinh``) are applied by the source, so cached values -- and every scaler fit -- are
post-transform.

**Online regimes (P7, §10.1).** ``VaeSource(..., unfreeze=prefixes)`` is trainable: only parameters under an
allowlisted module prefix may require gradients, and :class:`OnlineFeatures` (the ``nn.Module`` the classifier
registers as ``backbone``) keeps every other VAE module in ``eval()`` whatever Lightning's ``train()`` does, runs the
source on a batch's raw HDF5 rows and scales the result with the fold's frozen scaler. The frozen cache is still built
for these regimes: it carries the frames, the scaler, the priors and the frozen baseline unit.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn
from tqdm import tqdm

from hdf5_dataset.hdf5_dataset import DECIMATION, AttributeDict, CombinedHDF5Dataset, attribute_dict_collate
from teb_vae.classifier.cohort import SECONDS_PER_STEP, resolved_config_for
from teb_vae.classifier.config import REPO_ROOT, SourceCfg, resolve_path
from teb_vae.lag_attn.config import load_config
from teb_vae.lag_attn_rws.nets.controls import source_null_forward_outputs
from teb_vae.lag_attn_transformer_cfs.latent_pilot.config import software_record
from train.graph_models_utils import check_model_class, load_checkpoint_strict

TRANSFORMS = {"none": lambda x: x, "log1p": torch.log1p, "asinh": torch.asinh}
RAW_FIELDS = {"raw_fhr": "fhr", "raw_up": "up"}
IDENTITY_FIELDS = ("weight", "guid", "epoch")
#: Cohort columns that identify one (fold, split) occurrence of a stored segment.
INDEX_COLUMNS = ["fold", "split", "guid", "epoch_s", "source_file", "ds_index"]
#: Loader filters that decide who is in the cohort: inherited from a checkpoint, they are refused.
SHAPING_FILTERS = ("epoch_max", "cs_label", "bg_label", "allowed_guids")
#: Source files on every extraction path; VaeSource adds its VAE packages (§8.4).
CODE_FILES = (Path(__file__), Path(sys.modules[CombinedHDF5Dataset.__module__].__file__))


@dataclass(frozen=True)
class StepFeatures:
    """One batch of per-step features (§8.1).

    ``values`` is ``(B, T', C)`` float, ``step_mask`` ``(B, T')`` bool, ``attn`` ``(B, T', C_a)``
    attention-only cues or None. ``channels`` names the ``C`` value channels, then the ``C_a``
    attention channels.
    """

    values: torch.Tensor
    step_mask: torch.Tensor
    attn: Optional[torch.Tensor]
    channels: Tuple[str, ...]


def _sha256(path: Any) -> str:
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _code_sha256(files: Sequence[Path]) -> str:
    """Content digest of ``files``, keyed by their path relative to the checkout (any clone location):
    uncommitted and untracked edits change it, unlike the git SHA."""
    return _digest({Path(os.path.relpath(path, REPO_ROOT)).as_posix(): _sha256(path) for path in files})


def _vae_code_files() -> List[Path]:
    """Every ``*.py`` (tests aside) of each repo ``teb_vae.<pkg>`` package loaded so far.

    Called once the model is built, so it covers the family package and every sibling it imports.
    ``teb_vae.classifier`` is left out: of it, only ``sources.py`` (in :data:`CODE_FILES`) extracts.
    """
    roots = {Path(root).resolve() for name, module in list(sys.modules.items())
             if name.startswith("teb_vae.") and name.count(".") == 1
             and name != "teb_vae.classifier" for root in getattr(module, "__path__", ())}
    return [path for root in sorted(roots) if root.is_relative_to(REPO_ROOT)
            for path in sorted(root.rglob("*.py")) if "tests" not in path.relative_to(root).parts]


def _normalised(paths: Sequence[str], kwargs: Mapping[str, Any]) -> CombinedHDF5Dataset:
    """``CombinedHDF5Dataset`` over ``paths``; refused if its statistics did not load."""
    dataset = CombinedHDF5Dataset(list(paths), **kwargs)
    if not dataset.is_normalization_enabled():  # the loader only warns
        raise ValueError(f"normalisation statistics {kwargs['stats_path']} did not load")
    return dataset


def _names(name: str, width: int) -> List[str]:
    return [name] if width == 1 else [f"{name}[{i}]" for i in range(width)]


def _stats_record(stats_path: Any, *trims: Any) -> Dict[str, Any]:
    """L13: the stats file's ``trim_minutes`` equals every trim given; returns path and SHA-256."""
    path = resolve_path(stats_path)
    with h5py.File(path, "r") as handle:
        stored = handle.attrs.get("trim_minutes")
    if (stored is None or any(t is None for t in trims)
            or len({float(stored), *map(float, trims)}) != 1):
        raise ValueError(f"L13: trim_minutes disagree: stats file {path} has {stored}, loader / "
                         f"classifier use {list(trims)}. The trim places every validity boundary, "
                         f"so all of them must be equal.")
    return {"stats_path": str(path), "stats_sha256": _sha256(path), "trim_minutes": float(stored)}


def _finish(values: torch.Tensor, mask: torch.Tensor, attn: Optional[torch.Tensor],
            channels: Sequence[str], pool: int) -> StepFeatures:
    """Zero invalid steps, then masked-mean ``pool``-step windows (valid if any step is)."""
    values = torch.where(mask[..., None], values, 0.0)
    attn = None if attn is None else torch.where(mask[..., None], attn, 0.0)
    if pool > 1:
        batch, steps = mask.shape
        if steps % pool:
            raise ValueError(f"source.time_pool={pool} does not divide T={steps}")
        windows = mask.reshape(batch, steps // pool, pool)
        count = windows.sum(-1, keepdim=True).clamp_min(1)
        pooled = [None if x is None else x.reshape(batch, steps // pool, pool, -1).sum(2) / count
                  for x in (values, attn)]
        (values, attn), mask = pooled, windows.any(-1)
    return StepFeatures(values=values, step_mask=mask, attn=attn, channels=tuple(channels))


# ---- VaeSource (§8.2) --------------------------------------------------------------------------
def trainer_class(package: str) -> type:
    """The one trainer class ``teb_vae.<package>.trainer`` defines, with MODEL_CLS / TASK_CLS."""
    if package == "lag_attn":
        raise NotImplementedError("source.vae.package 'lag_attn' has other output keys (single z, "
                                  "te_lag_map) and is unsupported in v1 (SPEC §2.7)")
    module = importlib.import_module(f"teb_vae.{package}.trainer")
    found = [c for c in vars(module).values() if isinstance(c, type)
             and c.__module__ == module.__name__ and hasattr(c, "TASK_CLS")]
    if len(found) != 1:
        raise ValueError(f"teb_vae.{package}.trainer defines {len(found)} trainer classes with "
                         f"MODEL_CLS/TASK_CLS; expected exactly one")
    return found[0]


def task_parameters(task_cls: type) -> set:
    """Keyword names ``task_cls(...)`` accepts: its ``__init__``'s, and while an ``__init__`` takes ``**kwargs``, those
    of the next ``__init__`` up the MRO (the one it forwards to), e.g. cfs ``seed`` + the rws task's loss weights."""
    names = set()
    for klass in task_cls.__mro__:
        if "__init__" not in klass.__dict__:
            continue
        params = inspect.signature(klass.__dict__["__init__"]).parameters.values()
        names |= {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)} - {"self"}
        if not any(p.kind is p.VAR_KEYWORD for p in params):
            break
    return names


def load_task(checkpoint: Path, package: str, device: str) -> Tuple[Any, Dict[str, Any]]:
    """Rebuild a VAE task from its checkpoint, the same way for every supported family.

    Port of ``lag_attn_cfs/eval/probe.py:677 load_task``, which does not import at HEAD
    (``lag_attn_cfs/eval/binding.py`` imports a missing ``analyses.time_shift``). The binding is the
    family trainer's ``MODEL_CLS``/``TASK_CLS``, built with every checkpoint hyperparameter its constructor takes
    (:func:`task_parameters`; absent ones keep the class defaults), so the co-training ``L_vae`` is the pretraining
    objective (``lambda_ms``/``lambda_deriv``/``lambda_boundary`` included, §10.6); ``compile_model`` stays off (the
    feature and co-training hooks need the eager module). Returns ``(task in eval mode on device, blob)``.
    """
    trainer = trainer_class(package)
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    check_model_class(blob, trainer.MODEL_CLS.__name__)
    kwargs, hp = blob.get("model_kwargs"), blob.get("hyper_parameters")
    if not kwargs or not hp:
        raise RuntimeError(f"{checkpoint} carries no model_kwargs or hyper_parameters; the model "
                           f"cannot be rebuilt without guessing")
    names = task_parameters(trainer.TASK_CLS) - {"base_model", "model_kwargs", "compile_model"}  # eager: the hooks
    task = trainer.TASK_CLS(trainer.MODEL_CLS(**kwargs), model_kwargs=kwargs,
                            **{name: hp[name] for name in names if name in hp})
    if load_checkpoint_strict(model=task.orig_model, checkpoint=blob) is None:
        raise RuntimeError(f"could not align {checkpoint} into {trainer.MODEL_CLS.__name__}; "
                           f"refusing to extract from random weights")
    return task.to(device).eval(), blob


def checkpoint_loader_kwargs(resolved: Mapping[str, Any]) -> Dict[str, Any]:
    """``CombinedHDF5Dataset`` kwargs from a checkpoint's resolved config.

    ``latent_pilot/data.py:216 pilot_loader_config`` semantics: the run's own ``stat_path``,
    ``normalize_fields``, ``trim_minutes`` and ``load_fields`` (plus the identity fields); ``label``
    cleared; inherited cohort-shaping filters refused. ``epoch_min`` is cleared too: the cohort
    table decides membership and ``ds_index`` indexes the unfiltered shards. Ported rather than
    imported because the pilot also demands the ST/PH input fields, which e2e does not load.
    """
    dataset = resolved.get("dataset_config") or {}
    loader = dataset.get("dataloader_config") or {}
    kwargs = dict(loader.get("dataset_kwargs") or {})
    shaping = [name for name in SHAPING_FILTERS if kwargs.get(name) is not None]
    if shaping:
        raise ValueError(f"the checkpoint's resolved config carries cohort-shaping loader "
                         f"filter(s) {shaping}; remove them rather than have them cleared silently")
    fields = list(kwargs.get("load_fields") or [])
    return dict(kwargs, load_fields=fields + [f for f in IDENTITY_FIELDS if f not in fields],
                label=None, epoch_min=None, cache_size=0, pin_memory=False,
                stats_path=dataset.get("stat_path"),
                normalize_fields=loader.get("normalize_fields"))


def attn_summary(alpha: torch.Tensor, bins: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Per head: expected lag, entropy, lag-bin masses; ``(B, T, H, L) -> (B, T, H, 2 + n_bins)``.

    Bins must tile ``[0, L)`` from 0; the last one runs to ``L - 1`` whatever its stated end. The entropy reads
    ``alpha · log(max(alpha, tiny))``: the same value as ``xlogy``, but a finite gradient at the exact zeros entmax
    attention always has (``xlogy``'s backward is NaN there, which a trainable regime would spread into every weight).
    """
    lags = alpha.shape[-1]
    if [lo for lo, _ in bins] != [0] + [hi + 1 for _, hi in bins[:-1]] or bins[-1][0] >= lags:
        raise ValueError(f"source.vae.lag_bins {list(bins)} must tile [0, {lags}) contiguously")
    lag = torch.arange(lags, dtype=alpha.dtype, device=alpha.device)
    entropy = -(alpha * alpha.clamp_min(torch.finfo(alpha.dtype).tiny).log()).sum(-1)
    parts = [(alpha * lag).sum(-1), entropy]
    parts += [alpha[..., lo:hi + 1].sum(-1) for lo, hi in bins[:-1]]
    return torch.stack(parts + [alpha[..., bins[-1][0]:].sum(-1)], -1)


def _under(name: str, prefixes: Sequence[str]) -> bool:
    """``name`` (a dotted module or parameter name) is ``p`` or lies under ``p.`` for some prefix."""
    return any(name == p or name.startswith(p + ".") for p in prefixes)


class VaeSource:
    """A VAE's per-step outputs (§8.2). Frozen (``unfreeze`` empty): ``eval()`` and no gradient throughout.

    ``unfreeze`` (``train.unfreeze``) makes it trainable: exactly the parameters under those module prefixes may
    require gradients (a prefix matching nothing raises), and gradients flow only when grad mode is on. Which of them
    do at a given moment is the regime's business (LPFT switches them on at its stage 2); :meth:`check_frozen` is the
    pre-step guard. ``sample_z`` (``source.vae.sample_z_train``, ``frozen_online``): ``__call__(batch, sample_z=True)``
    replaces the ``mu_post`` and ``delta_mu`` values by a reparameterised draw of the posterior; the KL keys
    (``kld_per_dim``, ``kld_excess``) stay on the posterior means.

    ``kld_excess`` = ``kld_per_t`` minus the per-step KL of the posterior rebuilt against a zero source
    (``controls.source_null_forward_outputs``, one extra source encode, no gradient): the availability clock removed
    (§2.6). Families whose model has no ``encode_source_kv`` / ``u_stream`` input (``lag_attn_transformer_e2e``,
    ``lag_slot_transformer_cfs``) refuse it.
    """

    def __init__(self, cfg: SourceCfg, *, device: str = "cpu", unfreeze: Sequence[str] = (),
                 sample_z: bool = False) -> None:
        vae = cfg.vae
        checkpoint = resolve_path(vae.checkpoint)
        self.task, blob = load_task(checkpoint, vae.package, device)
        self.model = self.task.orig_model  # on ``device``; __call__ follows it wherever it moves (OnlineFeatures)
        self.keys, self.lag_bins = list(vae.keys), [tuple(b) for b in vae.lag_bins]
        nullable = ("u_stream" in inspect.signature(self.model.forward).parameters  # the source-null arm's input
                    and hasattr(self.model, "encode_source_kv"))
        if any(key.name == "kld_excess" for key in vae.keys) and not nullable:
            raise NotImplementedError(f"kld_excess needs a source-null forward (encode_source_kv over a u_stream "
                                      f"input); {type(self.model).__name__} ({vae.package}) has none")
        self.unfreeze, self.trainable, self.sample_z = tuple(unfreeze), bool(unfreeze), sample_z
        names = [name for name, _ in self.model.named_parameters()]
        unmatched = [p for p in self.unfreeze if not any(_under(n, [p]) for n in names)]
        if unmatched:
            raise ValueError(f"train.unfreeze prefix(es) {unmatched} match no parameter of "
                             f"{type(self.model).__name__}; e.g. {sorted({n.rsplit('.', 1)[0] for n in names})[:8]}")
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(_under(name, self.unfreeze))
        self.pool = cfg.time_pool
        self.dataset_kwargs = checkpoint_loader_kwargs(
            load_config(str(resolved_config_for(checkpoint))))
        # L13: stats file == checkpoint loader == classifier (source.hdf5.trim_minutes, CONTRACT).
        stats = _stats_record(self.dataset_kwargs["stats_path"],
                              self.dataset_kwargs.get("trim_minutes"), cfg.hdf5.trim_minutes)
        self.dataset_kwargs["stats_path"] = stats["stats_path"]  # the file fingerprinted, any cwd
        self.required_fields = tuple(self.dataset_kwargs["load_fields"])
        self.step_seconds = SECONDS_PER_STEP * cfg.time_pool
        self.warmup = int(self.model.warmup_period)
        supervised = getattr(self.model, "anchor_ceiling", self.model.geometry.t_valid)  # T - H
        causal_all = vae.step_support == "causal_all"
        self.ceiling = int(self.model.geometry.t if causal_all else supervised)
        self.fingerprint = {
            "kind": "vae", "package": vae.package, "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint), "model_class": blob.get("model_class"),
            "model_kwargs_sha256": _digest(blob["model_kwargs"]),
            "keys": [key.model_dump() for key in vae.keys], "lag_bins": self.lag_bins,
            "step_support": vae.step_support, "warmup": self.warmup, "ceiling": self.ceiling,
            "load_fields": list(self.required_fields), "time_pool": cfg.time_pool, **stats,
            "loader_sha256": _digest(self.dataset_kwargs),  # normalize_fields etc. change the features
            "code_sha256": _code_sha256([*CODE_FILES, *_vae_code_files()]),
        }
        logger.info(f"VaeSource {blob.get('model_class')} from {checkpoint}: steps "
                    f"[{self.warmup}, {self.ceiling}) ({vae.step_support}), keys "
                    f"{[k.name for k in vae.keys]}")

    def dataset(self, paths: Sequence[str]) -> CombinedHDF5Dataset:
        """Unfiltered dataset over ``paths`` under the checkpoint's loader contract."""
        return _normalised(paths, self.dataset_kwargs)

    def allowlist(self) -> List[Tuple[str, torch.nn.Parameter]]:
        """``(name, parameter)`` of every VAE parameter under ``train.unfreeze``, trainable now or not."""
        return [(n, p) for n, p in self.model.named_parameters() if _under(n, self.unfreeze)]

    def check_frozen(self) -> None:
        """The §10.1 pre-step guard (pilot ``check_pilot_mode``): no parameter outside the allowlist requires
        gradients, and every module outside it is in ``eval()`` (dropout off)."""
        leaked = [n for n, p in self.model.named_parameters() if p.requires_grad and not _under(n, self.unfreeze)]
        training = [n or "<root>" for n, m in self.model.named_modules() if m.training and not _under(n, self.unfreeze)]
        if leaked or training:
            raise RuntimeError(f"VAE outside train.unfreeze={list(self.unfreeze)}: parameter(s) requiring grad "
                               f"{leaked[:5]}, module(s) in training mode {training[:5]}")

    def _output(self, out: Mapping[str, torch.Tensor], name: str, slot: bool) -> torch.Tensor:
        if name == "delta_mu":
            return out["mu_post"] - out["mu_prior"]
        if name == "kld_per_dim":
            return out["kld_per_anchor_dim"] if slot else self.model.kld_tensor(
                mu_prior=out["mu_prior"], logvar_prior=out["logvar_prior"],
                mu_post=out["mu_post"], logvar_post=out["logvar_post"])
        if name == "kld_per_t" and slot:
            return out["kld_per_anchor"]
        if name == "attn_summary":
            return attn_summary(out["attn_weights"], self.lag_bins)
        if name not in out:
            raise KeyError(f"{name!r} is not an output of {type(self.model).__name__}; it returns "
                           f"{sorted(out)}")
        return out[name]

    def _channel_names(self, name: str, width: int) -> List[str]:
        if name != "attn_summary":
            return _names(name, width)
        stats = ["lag", "entropy"] + [f"bin{lo}" for lo, _ in self.lag_bins]
        return [f"attn_summary[h{h}.{s}]" for h in range(width // len(stats)) for s in stats]

    def __call__(self, batch: Any, *, sample_z: bool = False) -> StepFeatures:
        """Features of one collated batch; gradients reach the allowlist only when trainable and grad mode is on."""
        return self._run(batch, sample_z=sample_z)[0]

    def objective(self, batch: Any) -> Tuple[StepFeatures, torch.Tensor, Dict[str, Any]]:
        """Co-training (§10.6): ``(features, loss, metrics)`` of the VAE task's own ``compute_loss_and_metrics`` in its
        training geometry (tiled anchors; β as the task's hparams say), the features read off that same forward by a
        hook, so the model runs once. Every family's latents are dense on T whatever the geometry (only the decoder's
        anchor axis is tiled), so they are the features ``__call__`` gives; the slot family's are on the anchor axis,
        which a tiled forward leaves sparse, and :meth:`_run` refuses them."""
        return self._run(batch, stage="train")

    def _run(self, batch: Any, *, sample_z: bool = False, stage: Optional[str] = None
             ) -> Tuple[StepFeatures, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        device = next(self.model.parameters()).device
        batch = self.task.transfer_batch_to_device(batch, device, 0)
        loss = vae_metrics = None
        with torch.set_grad_enabled(self.trainable and torch.is_grad_enabled()):
            if stage is None:
                inputs = self.task._build_forward_inputs(batch)
                out = dict(self.model(*inputs))
            else:
                seen: List[Tuple[Any, Any]] = []
                handle = self.model.register_forward_hook(lambda _m, args, output: seen.append((args, output)))
                try:
                    loss, vae_metrics = self.task.compute_loss_and_metrics(batch, 0, stage)
                finally:
                    handle.remove()
                if len(seen) != 1:
                    raise RuntimeError(f"the VAE objective ran {len(seen)} forwards in one step; expected exactly 1")
                inputs, out = seen[0][0], dict(seen[0][1])
            # sample_z feeds the mu_post / delta_mu values only; the KL keys stay on the posterior means
            drawn = dict(out, mu_post=out["mu_post"] + torch.randn_like(out["mu_post"])
                         * (0.5 * out["logvar_post"]).exp()) if sample_z else out
            if any(key.name == "kld_excess" for key in self.keys):
                null = source_null_forward_outputs(self.model, out, inputs[2])
                out["kld_excess"] = out["kld_per_t"] - self.model.kld_tensor(
                    mu_prior=out["mu_prior"], logvar_prior=out["logvar_prior"], mu_post=null["mu_post"],
                    logvar_post=null["logvar_post"]).sum(-1)
            weight = batch["weight"]
            n, steps = weight.shape
            t = torch.arange(steps, device=weight.device)
            mask = (weight > 0) & (t >= self.warmup) & (t < self.ceiling)
            slot = "kld_per_anchor" in out  # lag_slot_transformer_cfs: latents on the anchor axis
            if slot:
                index, valid = out["anchor_index"], out["anchor_valid"].bool()
                if not (index.diff(dim=1)[valid[:, 1:]] == 1).all():
                    raise ValueError("slot family: anchors are not dense stride-1; cannot scatter")
                rows = torch.arange(n, device=index.device)[:, None].expand_as(index)
                at = (rows[valid], index[valid])
                on_grid = torch.zeros_like(mask)
                on_grid[at] = True
                mask &= on_grid
            parts: Dict[str, List[torch.Tensor]] = {"value": [], "attention": []}
            names: Dict[str, List[str]] = {"value": [], "attention": []}
            for key in self.keys:
                x = self._output(drawn if key.name in ("mu_post", "delta_mu") else out, key.name, slot).float()
                if slot and x.shape[1] != steps:  # anchor axis -> T; unscattered steps are masked
                    anchored, x = x[valid], x.new_zeros(n, steps, *x.shape[2:])
                    x[at] = anchored
                x = TRANSFORMS[key.transform](x.reshape(n, steps, -1))
                parts[key.role].append(x)
                names[key.role] += self._channel_names(key.name, x.shape[-1])
            values = (torch.cat(parts["value"], -1) if parts["value"]
                      else weight.new_zeros(n, steps, 0))
            attn = torch.cat(parts["attention"], -1) if parts["attention"] else None
        return _finish(values, mask, attn, names["value"] + names["attention"], self.pool), loss, vae_metrics


def _take(value: Any, index: torch.Tensor) -> Any:
    """Rows ``index`` of one collated field: a tensor, or a list (strings)."""
    return value[index.to(value.device)] if torch.is_tensor(value) else [value[i] for i in index.tolist()]


#: ``train/<name>`` of a co-training step (§10.10.3) <- the VAE objective's metric it reads.
VAE_METRICS = {"vae_kld": "source_conditioned_kl_raw", "vae_nll": "nll_full_block"}


class OnlineFeatures(nn.Module):
    """A :class:`VaeSource` run inside the classifier (the online regimes, §10.1): registered on the Lightning task
    as ``backbone``, so the device moves, the optimizer groups, EMA and ``best.ckpt`` all see the VAE (``vae.*``) and
    the fold's frozen scaler (persistent buffers). The VAE's Lightning task stays outside the module tree.

    ``forward(batch)`` runs the source over ``batch["vae"]`` (the collated raw HDF5 rows of the batch's real
    segments, in ``seg_mask`` order; ``data.OnlineReader``) in chunks of ``chunk`` segments, scales ``[values ||
    attn]`` exactly as ``UnitData.features`` does, and returns a shallow copy of ``batch`` whose ``x`` / ``attn`` hold
    the online features at the real positions. The step mask never depends on the weights, so it must equal the
    cached one; a difference raises (the rows are not the cached segments).

    ``train(mode)`` keeps every module outside the allowlist in ``eval()``: Lightning calls ``train()`` on the whole
    task every epoch, and dropout in a frozen module would change the features the head was fitted on.

    ``cotrain`` (``train.cotrain``, §10.6; training steps only): the rows run through :meth:`VaeSource.objective`, and
    the returned batch adds ``vae_loss`` (the VAE objective, a row-weighted mean over the rows that train) and
    ``vae_metrics`` (:data:`VAE_METRICS` + ``vae_total_loss``). Under ``grad_segments: last_k:<k>`` only each GUID's
    last k segments train (and enter ``vae_loss``); the others run the plain forward under ``no_grad`` (``ponytail:``
    a crude memory cap; gradient checkpointing if long GUIDs matter). ``detach_head_input``: the head reads sg(features).
    :meth:`l2sp` is ``‖θ - θ₀‖²`` over the allowlist, θ₀ the pretrained weights (non-persistent buffers).
    """

    def __init__(self, source: VaeSource, scaler: "Scaler", *, n_values: int, chunk: int = 32,
                 cotrain: Any = None) -> None:
        """``n_values``: the kept value channels (``UnitData.n_values``); the kept rest are attention cues."""
        super().__init__()
        self.source, self.vae, self.chunk, self.n_values = source, source.model, int(chunk), int(n_values)
        self.channels, keep = tuple(scaler.channels), np.asarray(scaler.keep, dtype=bool)
        self.keep = torch.as_tensor(np.flatnonzero(keep))  # not a buffer: EMA lerps every buffer, and floats only
        self.register_buffer("center", torch.as_tensor(np.asarray(scaler.center, dtype=np.float32)[keep]))
        self.register_buffer("scale", torch.as_tensor(np.asarray(scaler.scale, dtype=np.float32)[keep]))
        self.cotrain = cotrain
        if cotrain is not None:
            self.last_k = None if cotrain.grad_segments == "all" else int(cotrain.grad_segments.split(":")[1])
            for i, (_, p) in enumerate(source.allowlist()):
                self.register_buffer(f"theta0_{i}", p.detach().clone(), persistent=False)
        # Train mode, as a fresh module is: Lightning (>= 2.2) keeps every submodule's mode at fit start instead of
        # calling train(), so a backbone built in eval would train in eval (no co-training objective, no sample_z).
        # Scoring (_score) and validation switch it to eval themselves.
        self.train()

    def l2sp(self) -> torch.Tensor:
        """L2-SP (§10.6): ``Σ ‖θ - θ₀‖²`` over the allowlisted VAE parameters."""
        return sum(((p - getattr(self, f"theta0_{i}")) ** 2).sum() for i, (_, p) in enumerate(self.source.allowlist()))

    def _trains(self, real: Optional[torch.Tensor], n: int) -> torch.Tensor:
        """(n,) bool on the host: which rows (``seg_mask`` order) run the co-training objective."""
        if self.last_k is None or real is None:  # segment scope has no positions (config refuses last_k there)
            return torch.ones(n, dtype=torch.bool)
        real = real.cpu()
        from_end = real.sum(1, keepdim=True) - 1 - torch.arange(real.shape[1])
        return (from_end < self.last_k)[real]

    def train(self, mode: bool = True) -> "OnlineFeatures":
        """The allowlisted modules train while their parameters do (LPFT stage 1 keeps them in eval too)."""
        super().train(mode)
        self.vae.eval()
        if mode and any(p.requires_grad for _, p in self.source.allowlist()):
            for name, module in self.vae.named_modules():
                if _under(name, self.source.unfreeze):
                    module.train()
        return self

    def forward(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        rows, real = batch["vae"], batch.get("seg_mask")
        n = len(rows["weight"])
        joint = self.training and self.cotrain is not None
        trains = self._trains(real, n) if joint else torch.ones(n, dtype=torch.bool)
        order, parts, losses = [], [], []
        for learns, group in ((True, trains), (False, ~trains)):
            index = torch.nonzero(group).flatten()
            for i in range(0, len(index), self.chunk):
                sub = index[i:i + self.chunk]
                chunk = AttributeDict({k: _take(v, sub) for k, v in rows.items()})  # the VAE tasks read attributes
                if joint and learns:
                    features, loss, m = self.source.objective(chunk)
                    losses.append((len(sub), loss, m))
                elif joint:
                    with torch.no_grad():
                        features = self.source(chunk)
                else:
                    features = self.source(chunk, sample_z=self.training and self.source.sample_z)
                order.append(sub)
                parts.append(features)
        if parts[0].channels != self.channels:
            raise ValueError(f"online channels {parts[0].channels} differ from the scaler's {self.channels}")
        back = torch.argsort(torch.cat(order))  # the chunks' rows back in seg_mask order
        mask = torch.cat([p.step_mask for p in parts])[back.to(parts[0].step_mask.device)]
        cached = batch["step_mask"] if real is None else batch["step_mask"][real]
        if not torch.equal(mask, cached.to(mask.device)):
            raise ValueError("online step mask differs from the cached one: the HDF5 rows are not the cached segments")
        stacked = torch.cat([torch.cat([p.values] + ([p.attn] if p.attn is not None else []), -1) for p in parts])
        z = (stacked[back.to(stacked.device)][..., self.keep.to(stacked.device)] - self.center) / self.scale
        z = torch.where(mask[..., None], z, 0.0)
        out = dict(batch)
        if joint:
            if self.cotrain.detach_head_input:
                z = z.detach()
            w = [k / sum(k for k, _, _ in losses) for k, _, _ in losses]
            out["vae_loss"] = sum(wi * loss for wi, (_, loss, _) in zip(w, losses))
            out["vae_metrics"] = {"vae_total_loss": out["vae_loss"].detach(),
                                  **{name: sum(wi * m[key].detach() for wi, (_, _, m) in zip(w, losses))
                                     for name, key in VAE_METRICS.items()}}
        for key, part in (("x", z[..., :self.n_values]), ("attn", z[..., self.n_values:])):
            if key in batch:
                if real is None:
                    out[key] = part.to(batch[key].dtype)
                else:
                    out[key] = batch[key].clone()
                    out[key][real] = part.to(batch[key].dtype)
        return out


# ---- Hdf5Source (§8.3) -------------------------------------------------------------------------
class Hdf5Source:
    """Stored ST/PH/raw streams (§8.3), normalised by ``stats_path``; cold causal cells zeroed.

    ``probe_shard`` is any shard of the tree: channel counts and warm-ups are read from it.
    """

    trainable = False

    def __init__(self, cfg: SourceCfg, probe_shard: str) -> None:
        hdf5 = cfg.hdf5
        self.fields, self.pool = list(hdf5.fields), cfg.time_pool
        self.step_seconds = SECONDS_PER_STEP * cfg.time_pool
        stats = _stats_record(hdf5.stats_path, hdf5.trim_minutes)
        self.required_fields = tuple(dict.fromkeys(
            [RAW_FIELDS.get(f, f) for f in self.fields] + list(IDENTITY_FIELDS)))
        self.dataset_kwargs: Dict[str, Any] = dict(
            load_fields=self.required_fields, stats_path=stats["stats_path"],
            trim_minutes=hdf5.trim_minutes, cache_size=0, pin_memory=False)
        probe = _normalised([probe_shard], self.dataset_kwargs)
        warmup = probe.causal_warmup_steps  # rebased for the trim; None for two-sided builds
        self.causal = warmup is not None
        self.dataset_kwargs["emit_validity_mask"] = self.causal
        sample = probe[0]
        missing = [f for f in self.required_fields if f not in sample]
        if missing:
            raise ValueError(f"{probe_shard} stores no {missing}; source.hdf5.fields asks for them")
        self.widths = {f: DECIMATION if f in RAW_FIELDS else int(sample[f].shape[-1])
                       for f in self.fields}
        coefficients = [f for f in self.fields if f not in RAW_FIELDS]
        auto = max((int(warmup[f].max()) for f in coefficients), default=0) if self.causal else 0
        self.min_step = auto if hdf5.min_step == "auto" else int(hdf5.min_step or 0)
        self.channels = [name for f in self.fields for name in _names(f, self.widths[f])]
        self.fingerprint = {"kind": "hdf5", "fields": self.fields, "widths": self.widths,
                            "causal": self.causal, "min_step": self.min_step,
                            "time_pool": cfg.time_pool, **stats,
                            "code_sha256": _code_sha256(CODE_FILES)}
        logger.info(f"Hdf5Source {self.widths} (causal={self.causal}), steps >= {self.min_step}")

    def dataset(self, paths: Sequence[str]) -> CombinedHDF5Dataset:
        """Unfiltered dataset over ``paths``, normalised, with validity masks on causal builds."""
        return _normalised(paths, self.dataset_kwargs)

    def __call__(self, batch: Any) -> StepFeatures:
        weight = batch["weight"]
        n, steps = weight.shape
        parts, warm = [], torch.zeros_like(weight, dtype=torch.bool)
        for field in self.fields:
            if field in RAW_FIELDS:
                x, ok = batch[RAW_FIELDS[field]].reshape(n, steps, DECIMATION), None
            else:
                x, ok = batch[field], batch.get(f"{field}_valid")
            if ok is None:
                warm[:] = True
            else:
                x, warm = torch.where(ok, x, 0.0), warm | ok.any(-1)
            parts.append(x.float())
        mask = (weight > 0) & warm & (torch.arange(steps) >= self.min_step)
        return _finish(torch.cat(parts, -1), mask, None, self.channels, self.pool)


def make_source(cfg: SourceCfg, probe_shard: str, *, device: str = "cpu") -> Any:
    """The configured source; ``probe_shard`` (any shard of the tree) is read by Hdf5Source only."""
    return VaeSource(cfg, device=device) if cfg.kind == "vae" else Hdf5Source(cfg, probe_shard)


# ---- cache (§8.4) ------------------------------------------------------------------------------
#: Fingerprint fields kept for information only: neither in the cache key nor checked (``code_sha256`` covers the code).
INFO_FIELDS = {"code_revision"}


def _check_fingerprint(stored: Mapping[str, Any], expected: Mapping[str, Any], where: Any) -> None:
    differences = {key: (stored.get(key), expected.get(key))
                   for key in sorted((set(stored) | set(expected)) - INFO_FIELDS)
                   if stored.get(key) != expected.get(key)}
    if differences:
        raise ValueError(f"feature cache {where} was built under another fingerprint; refusing to "
                         f"reuse it. Differences (stored, expected): {differences}")


def _cache_bytes(directory: Path) -> int:
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


def extract(source: Any, segments: pd.DataFrame, shards: Mapping[Tuple[int, str], Sequence[str]], *,
            cache_root: Any, dtype: str = "float16", batch_size: int = 64) -> Dict[str, Any]:
    """Run ``source`` once per unique ``(guid, epoch_s)`` of the retained cohort rows (§8.4).

    Args:
        source: A :class:`VaeSource` or :class:`Hdf5Source`.
        segments: The cohort segment table (all folds and splits); excluded rows are skipped.
        shards: ``{(fold, split): shard paths}`` exactly as the cohort stage read them, so
            ``ds_index`` addresses the same rows. Each read is checked against ``(guid, epoch_s)``.
        cache_root: Parent of the ``<fingerprint hash>/`` directory.
        dtype: Storage dtype of ``values``/``attn`` (compute is fp32).
        batch_size: Segments per forward; also the resume granularity.

    Returns:
        The manifest record: ``cache_dir``, ``fingerprint``, ``fingerprint_hash``, counts,
        ``seconds``, ``segments_per_s`` and ``bytes``.

    Raises:
        ValueError: On a fingerprint mismatch in an existing directory, a row that does not read
            back as its ``(guid, epoch_s)``, or non-finite features.
    """
    index = (segments.loc[~segments["excluded"].astype(bool), INDEX_COLUMNS]
             .sort_values(["fold", "split", "ds_index"], ignore_index=True))
    unique = index.drop_duplicates(["guid", "epoch_s"], ignore_index=True)
    unique["row"] = np.arange(len(unique))
    index = index.merge(unique[["guid", "epoch_s", "row"]], on=["guid", "epoch_s"])
    paths = sorted({p for key in {(int(f), str(s)) for f, s in zip(index["fold"], index["split"])}
                    for p in shards[key]})
    fingerprint = json.loads(json.dumps({
        **source.fingerprint, "cache_dtype": dtype, "code_revision": software_record()["revision"],
        "shards": [{"path": p, "size": os.stat(p).st_size, "mtime": os.stat(p).st_mtime}
                   for p in paths],
        "index_sha256": hashlib.sha256(
            index[INDEX_COLUMNS].to_csv(index=False).encode()).hexdigest(),
    }, default=str))
    key = _digest({k: v for k, v in fingerprint.items() if k not in INFO_FIELDS})[:16]
    directory = resolve_path(cache_root) / key
    directory.mkdir(parents=True, exist_ok=True)
    stored = directory / "fingerprint.json"
    if stored.is_file():
        _check_fingerprint(json.loads(stored.read_text()), fingerprint, directory)
    else:
        index.to_parquet(directory / "index.parquet", index=False)
        stored.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))

    datasets: Dict[Tuple[int, str], CombinedHDF5Dataset] = {}
    started = time.perf_counter()
    with h5py.File(directory / "features.h5", "a") as h5:
        done = h5.require_dataset("done", (len(unique),), dtype=bool, fillvalue=False)
        todo = np.flatnonzero(~done[:])
        # ponytail: single-process per-sample h5 reads; add DataLoader workers if I/O-bound.
        for start in tqdm(range(0, len(todo), batch_size), desc="extract", unit="batch"):
            rows = todo[start:start + batch_size]
            part = unique.iloc[rows]
            chunks = []
            for (fold, split), group in part.groupby(["fold", "split"], sort=False):
                shard_key = (int(fold), str(split))
                if shard_key not in datasets:
                    datasets[shard_key] = source.dataset(shards[shard_key])
                dataset = datasets[shard_key]
                batch = attribute_dict_collate([dataset[int(i)] for i in group["ds_index"]])
                if (list(batch["guid"]) != group["guid"].tolist() or not np.array_equal(
                        batch["epoch"].double().numpy(), group["epoch_s"].to_numpy(float))):
                    raise ValueError(f"row alignment: ds_index in {shard_key} does not read back "
                                     f"as the cohort's (guid, epoch_s); rebuild the cohort stage")
                chunks.append(source(batch))
            values = torch.cat([c.values for c in chunks]).cpu().numpy().astype(dtype)
            attn = (None if chunks[0].attn is None
                    else torch.cat([c.attn for c in chunks]).cpu().numpy().astype(dtype))
            if not np.isfinite(values).all() or (attn is not None and not np.isfinite(attn).all()):
                raise ValueError(f"non-finite features in cache rows {rows[0]}..{rows[-1]} "
                                 f"(overflow in {dtype}?)")
            if "values" not in h5:
                h5.create_dataset("values", (len(unique), *values.shape[1:]), dtype=dtype)
                h5.create_dataset("step_mask", (len(unique), values.shape[1]), dtype=bool)
                if attn is not None:
                    h5.create_dataset("attn", (len(unique), *attn.shape[1:]), dtype=dtype)
                h5.attrs["channels"] = json.dumps(list(chunks[0].channels))
            h5["values"][rows] = values
            h5["step_mask"][rows] = torch.cat([c.step_mask for c in chunks]).cpu().numpy()
            if attn is not None:
                h5["attn"][rows] = attn
            done[rows] = True
            h5.flush()
    seconds = time.perf_counter() - started
    record = {"cache_dir": str(directory), "fingerprint_hash": key, "fingerprint": fingerprint,
              "n_unique": int(len(unique)), "n_rows": int(len(index)),
              "n_extracted": int(len(todo)), "seconds": round(seconds, 3),
              "segments_per_s": round(len(todo) / seconds, 2) if len(todo) else None,
              "bytes": _cache_bytes(directory)}
    logger.info(f"extract: {record['n_extracted']} of {record['n_unique']} unique segments "
                f"({record['n_rows']} fold x split rows) in {seconds:.1f} s = "
                f"{record['segments_per_s']} segments/s; cache {record['bytes'] / 1e6:.1f} MB at "
                f"{directory}")
    return record


def open_cache(cache_dir: Any, expected: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """The cache index (one row per fold x split occurrence; ``row`` addresses the arrays).

    Raises:
        ValueError: If the stored fingerprint differs from ``expected`` or extraction is unfinished.
    """
    directory = Path(cache_dir)
    if expected is not None:
        stored = json.loads((directory / "fingerprint.json").read_text())
        _check_fingerprint(stored, expected, directory)
    with h5py.File(directory / "features.h5", "r") as h5:
        if not h5["done"][:].all():
            raise ValueError(f"feature cache {directory} is incomplete; rerun --stage extract")
    return pd.read_parquet(directory / "index.parquet")


def read_rows(cache_dir: Any, rows: Sequence[int]) -> StepFeatures:
    """Cached features for index ``row`` values (any order, repeats allowed), float32 / bool."""
    wanted = np.asarray(rows, dtype=np.int64)
    unique, inverse = np.unique(wanted, return_inverse=True)
    with h5py.File(Path(cache_dir) / "features.h5", "r") as h5:
        read = {name: h5[name][unique][inverse]
                for name in ("values", "step_mask", "attn") if name in h5}
        channels = tuple(json.loads(h5.attrs["channels"]))
    attn = read.get("attn")
    return StepFeatures(values=torch.from_numpy(read["values"].astype(np.float32)),
                        step_mask=torch.from_numpy(read["step_mask"]),
                        attn=None if attn is None else torch.from_numpy(attn.astype(np.float32)),
                        channels=channels)


# ---- scaler (§8.5) -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Scaler:
    """Per-channel ``(x - center) / scale`` over the kept channels; zero-variance ones dropped."""

    channels: Tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    keep: np.ndarray
    record: Dict[str, Any]

    def apply(self, values: Any) -> np.ndarray:
        """``(..., C) -> (..., C_kept)``."""
        values = np.asarray(values)
        if values.shape[-1] != len(self.channels):
            raise ValueError(f"scaler fitted on {len(self.channels)} channels, got "
                             f"{values.shape[-1]}")
        return (values[..., self.keep] - self.center[self.keep]) / self.scale[self.keep]

    def save(self, path: Any) -> None:
        payload = {"channels": list(self.channels), "center": self.center.tolist(),
                   "scale": self.scale.tolist(), "keep": self.keep.tolist(), "record": self.record}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Any) -> "Scaler":
        payload = json.loads(Path(path).read_text())
        return cls(channels=tuple(payload["channels"]), center=np.asarray(payload["center"]),
                   scale=np.asarray(payload["scale"]), keep=np.asarray(payload["keep"], dtype=bool),
                   record=payload["record"])


def fit_scaler(index: pd.DataFrame, values: Any, step_mask: Any, channels: Sequence[str], *,
               minimum_scale: float = 1e-3, relative_floor: float = 0.1) -> Scaler:
    """Fit the per-channel scaler on one fold's **train** rows (§8.5, L4).

    Generalises ``latent_pilot/extract.py:592 fit_scaler`` + ``_hierarchical_moments`` (not
    importable at HEAD, and keyed to one latent) to any stream: moments are means over valid steps
    within a segment, then over segments within a GUID, then over GUIDs equally. Scales are floored
    at ``max(minimum_scale, relative_floor * median positive std)``; a channel constant over every
    valid train step is dropped with a warning and recorded.

    Args:
        index: One row per segment, aligned with ``values``; needs ``split`` (all ``train``),
            ``guid`` and, if present, a single ``fold``.
        values: ``(N, T', C)``, post-transform (the source applies it).
        step_mask: ``(N, T')`` bool.
        channels: The ``C`` channel names.

    Raises:
        ValueError: On a non-train row, several folds, or no varying channel at all.
    """
    splits = sorted(set(index["split"].astype(str)))
    if splits != ["train"]:
        raise ValueError(f"L4: the scaler fits on the train split only; got {splits}")
    if "fold" in index and index["fold"].nunique() != 1:
        raise ValueError(f"the scaler is fitted per fold; got {sorted(index['fold'].unique())}")
    step_mask = np.asarray(step_mask, dtype=bool)
    sums, squares, lo, hi = [], [], np.inf, -np.inf
    for start in range(0, len(step_mask), 1024):  # float64 in chunks: the train fold is large
        x = np.asarray(values[start:start + 1024], dtype=np.float64)
        w = step_mask[start:start + 1024]
        sums.append(np.einsum("ntc,nt->nc", x, w.astype(np.float64)))
        squares.append(np.einsum("ntc,nt->nc", x * x, w.astype(np.float64)))
        lo = np.minimum(lo, np.where(w[..., None], x, np.inf).min((0, 1)))
        hi = np.maximum(hi, np.where(w[..., None], x, -np.inf).max((0, 1)))
    count = step_mask.sum(1)
    used = count > 0
    guids = index["guid"].astype(str).to_numpy()[used]
    first, second = (pd.DataFrame(np.concatenate(s)[used] / count[used, None]).groupby(guids).mean()
                     .mean().to_numpy() for s in (sums, squares))
    std = np.sqrt(np.maximum(second - first ** 2, 0.0))
    keep = hi > lo
    positive = std[keep & (std > 0)]
    if not positive.size:
        raise ValueError("no channel varies over the train split's valid steps")
    floor = max(minimum_scale, relative_floor * float(np.median(positive)))
    dropped = [name for name, k in zip(channels, keep) if not k]
    if dropped:
        logger.warning(f"scaler: dropping {len(dropped)} zero-variance channel(s): {dropped}")
    fold = int(index["fold"].iloc[0]) if "fold" in index else None
    record = {"population": "train", "fold": fold,
              "hierarchy": "steps within segments, segments within GUIDs, GUIDs equally",
              "n_guids": int(len(set(guids))), "n_segments": int(used.sum()),
              "n_steps": int(count.sum()), "floor": floor,
              "n_at_floor": int((keep & (std < floor)).sum()), "dropped": dropped}
    return Scaler(channels=tuple(channels), center=first, scale=np.maximum(std, floor), keep=keep,
                  record=record)
