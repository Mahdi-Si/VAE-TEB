import contextlib
import os
import warnings
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from loguru import logger


def _extract_state_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, (dict, OrderedDict)):
        if obj and all(isinstance(v, (torch.Tensor, torch.nn.Parameter)) for v in obj.values()):
            return obj
        for key in (
            "state_dict",
            "model_state_dict",
            "pytorch_model_state_dict",
            "model_state_dict",
            "model",
            "module",
            "network",
            "net",
            "weights",
            "state",
        ):
            if key in obj:
                extracted = _extract_state_dict(obj[key])
                if extracted is not None:
                    return extracted
        return None
    if hasattr(obj, "state_dict"):
        with contextlib.suppress(Exception):
            return obj.state_dict()
    return None


def _clean_state_dict(sd):
    cleaned = OrderedDict()
    prefixes = (
        "model._orig_mod.",
        "model._orig_model.",
        "_orig_model.",
        "_orig_mod.",
        "model.model.",
        "model.module.",
        "model.seqvae_teb_model.",
        "model.seqvae_model.",
        "seqvae_teb_model.",
        "seqvae_model.",
        "lightning_module.",
        "pytorch_model.",
        "module.",
        "network.",
        "net.",
        "model.",
    )
    for key, value in sd.items():
        if not isinstance(value, (torch.Tensor, torch.nn.Parameter)):
            continue
        new_key = key
        while True:
            matched = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    matched = True
                    break
            if not matched:
                break
        if not new_key:
            continue
        cleaned[new_key] = value
    return cleaned


def _normalize_checkpoint_state_dict(state_dict):
    normalized = OrderedDict()
    renamed = False
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in (
            ('model._orig_mod.', 'model.'),
            ('model._orig_model.', 'model.'),
            ('_orig_model.', ''),
            ('_orig_mod.', ''),
            ('model.model.', 'model.'),
        ):
            if new_key.startswith(old_prefix):
                new_key = new_prefix + new_key[len(old_prefix):]
                renamed = True
                break
        normalized[new_key] = value
    return normalized, renamed


def load_checkpoint_torch(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    state_dict = OrderedDict(state_dict)
    normalized_state_dict, renamed = _normalize_checkpoint_state_dict(state_dict)
    cleaned_state_dict = _clean_state_dict(normalized_state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
    if unexpected_keys:
            logger.warning(f"Unexpected keys while loading SeqVaeTeb checkpoint: {sorted(unexpected_keys)}")
    if missing_keys:
        logger.warning(f"Missing keys while loading SeqVaeTeb checkpoint: {sorted(missing_keys)}")
    else:
        logger.info("Successfully loaded SeqVaeTeb weights.")
    return model


_DEFAULT_MODULE_ATTR_CANDIDATES: Sequence[str] = (
    "model",
    "module",
    "_orig_mod",
    "wrapped_module",
    "pytorch_model",
    "lightning_module",
    "base_model",
    "network",
    "net",
)


def _discover_candidate_modules(
    root: Optional[Any],
    extra_attr_names: Optional[Iterable[str]] = None,
) -> List[torch.nn.Module]:
    if root is None:
        return []
    attr_names: Tuple[str, ...] = tuple(
        dict.fromkeys(
            list(_DEFAULT_MODULE_ATTR_CANDIDATES)
            + list(extra_attr_names or [])
        )
    )
    queue: List[Any] = [root]
    resolved: List[torch.nn.Module] = []
    visited = set()
    while queue:
        current = queue.pop(0)
        identifier = id(current)
        if identifier in visited:
            continue
        visited.add(identifier)
        if isinstance(current, torch.nn.Module):
            resolved.append(current)
        for attr_name in attr_names:
            nested = getattr(current, attr_name, None)
            if isinstance(nested, torch.nn.Module):
                queue.append(nested)
    return resolved


def _format_key_sample(keys: Sequence[str], limit: int = 8) -> str:
    if not keys:
        return ""
    subset = list(keys)
    subset.sort()
    head = subset[:limit]
    formatted = ", ".join(head)
    if len(subset) > limit:
        formatted = f"{formatted}, ... (+{len(subset) - limit} more)"
    return formatted


def _format_mismatched_entries(entries: Dict[str, str], limit: int = 5) -> str:
    if not entries:
        return ""
    items = list(entries.items())
    items.sort(key=lambda item: item[0])
    head = items[:limit]
    formatted = "; ".join(f"{key}: {reason}" for key, reason in head)
    if len(items) > limit:
        formatted = f"{formatted}; ... (+{len(items) - limit} more)"
    return formatted


def _evaluate_state_alignment(
    module: torch.nn.Module,
    checkpoint_state: OrderedDict,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    module_state = module.state_dict()
    module_keys = list(module_state.keys())
    checkpoint_keys = list(checkpoint_state.keys())
    missing = [key for key in module_keys if key not in checkpoint_state]
    unexpected = [key for key in checkpoint_keys if key not in module_state]
    mismatched: Dict[str, str] = {}
    for key in module_keys:
        if key not in checkpoint_state:
            continue
        module_tensor = module_state[key]
        checkpoint_tensor = checkpoint_state[key]
        module_shape = tuple(module_tensor.shape)
        checkpoint_shape = tuple(checkpoint_tensor.shape)
        if module_shape != checkpoint_shape:
            mismatched[key] = f"expected shape {module_shape} but found {checkpoint_shape}"
            continue
        module_dtype = getattr(module_tensor, "dtype", None)
        checkpoint_dtype = getattr(checkpoint_tensor, "dtype", None)
        if module_dtype is not None and checkpoint_dtype is not None and module_dtype != checkpoint_dtype:
            mismatched[key] = f"expected dtype {module_dtype} but found {checkpoint_dtype}"
    return missing, unexpected, mismatched


def _prepare_checkpoint_state_dict(
    checkpoint: Any,
    map_location: str = "cpu",
) -> Optional[OrderedDict]:
    checkpoint_obj = checkpoint
    if isinstance(checkpoint, (str, os.PathLike)):
        checkpoint_path = os.fspath(checkpoint)
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found at {checkpoint_path}")
            return None
        logger.info(f"Reading checkpoint from {checkpoint_path}")
        checkpoint_obj = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = _extract_state_dict(checkpoint_obj)
    if state_dict is None:
        logger.error("Unable to extract a state_dict from the provided checkpoint reference")
        return None
    ordered_state = OrderedDict(state_dict)
    normalized_state_dict, renamed = _normalize_checkpoint_state_dict(ordered_state)
    cleaned_state_dict = _clean_state_dict(normalized_state_dict)
    if not cleaned_state_dict:
        logger.error("Checkpoint state_dict was empty after normalization and cleaning steps")
        return None
    if renamed:
        logger.info("Checkpoint keys were normalized prior to loading")
    return cleaned_state_dict


def load_checkpoint_strict(
    model: Optional[torch.nn.Module],
    checkpoint: Any,
    *,
    map_location: str = "cpu",
    module_attr_names: Optional[Iterable[str]] = None,
) -> Optional[torch.nn.Module]:
    """
    Load checkpoint weights into the provided model or any wrapped nn.Module it contains.

    Args:
        model: The target PyTorch module, LightningModule, compiled module, or wrapper.
        checkpoint: Path to a checkpoint, an object containing a state_dict, or a raw state_dict.
        map_location: torch.load map_location argument for file-based checkpoints.
        module_attr_names: Optional iterable of attribute names to scan when hunting
            for nested modules (extends the default common wrappers).

    Returns:
        The original model after loading on success, otherwise None.
    """
    if model is None:
        logger.error("A valid model instance is required for checkpoint loading")
        return None
    checkpoint_state = _prepare_checkpoint_state_dict(checkpoint, map_location=map_location)
    if checkpoint_state is None:
        return None
    candidate_modules = _discover_candidate_modules(model, module_attr_names)
    if not candidate_modules:
        logger.error("No torch.nn.Module instances were found within the provided wrapper")
        return None
    for candidate in candidate_modules:
        missing, unexpected, mismatched = _evaluate_state_alignment(candidate, checkpoint_state)
        if missing or unexpected or mismatched:
            candidate_name = candidate.__class__.__name__
            if missing:
                logger.warning(f"[{candidate_name}] Missing keys: {_format_key_sample(missing)}")
            if unexpected:
                logger.warning(f"[{candidate_name}] Unexpected keys: {_format_key_sample(unexpected)}")
            if mismatched:
                logger.warning(f"[{candidate_name}] Shape/dtype mismatches: {_format_mismatched_entries(mismatched)}")
            continue
        candidate.load_state_dict(checkpoint_state, strict=True)
        logger.info(f"Checkpoint successfully loaded into {candidate.__class__.__name__}")
        return model
    logger.error("Checkpoint keys did not align with any discovered module")
    return None


def check_model_class(ckpt: Any, active_cls_name: str) -> None:
    r"""Guard that a checkpoint's ``model_class`` matches the class about to load it.

    Reads the field :meth:`train.pl_model_base.LightningModelBase.on_save_checkpoint` stamps, and
    is the only enforcement point for it. Run this on the raw checkpoint dict **before** any
    ``Model(**model_kwargs)`` reconstruction: model constructors in this repo are keyword-only
    with no ``**kwargs``, so loading one version's ``model_kwargs`` into another would otherwise
    raise a cryptic ``TypeError`` at construction. This guard fails first, with a message naming
    both classes.

    A checkpoint with no ``model_class`` warns rather than raises: checkpoints predating the stamp
    are legitimate, and the rebuild still fails loudly if the kwargs are genuinely incompatible.

    Args:
        ckpt: The deserialised checkpoint. Non-dicts are skipped, so a bare state-dict is not an
            error -- it simply carries no claim to check.
        active_cls_name: ``__name__`` of the class that is about to load it.

    Raises:
        ValueError: If the checkpoint records a ``model_class`` differing from
            ``active_cls_name``.
    """
    if not isinstance(ckpt, dict):
        return
    stored = ckpt.get("model_class")
    if stored is None:
        warnings.warn(
            "checkpoint has no 'model_class' field (pre-guard checkpoint); "
            f"assuming it matches the active class {active_cls_name!r}. The "
            "rebuild will still fail loudly if the constructor kwargs are "
            "incompatible.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    if str(stored) != str(active_cls_name):
        raise ValueError(
            f"checkpoint model_class={stored!r} does not match the active model "
            f"class {active_cls_name!r}. Load this checkpoint with the class that "
            f"wrote it, or point the run at a checkpoint written by "
            f"{active_cls_name!r}."
        )


def denormalize_signal_data(normalized_data: torch.Tensor, field_name: str, normalization_stats: dict) -> torch.Tensor:
    """
    Denormalize FHR or UP signal data using normalization statistics.
    
    Args:
        normalized_data: Normalized tensor data (shape: any)
        field_name: Name of the field ('fhr' or 'up')
        normalization_stats: Dictionary containing normalization statistics
        
    Returns:
        Denormalized tensor data
    """
    if field_name not in normalization_stats:
        logger.warning(f"No normalization stats found for field '{field_name}'. Returning data as-is.")
        return normalized_data
    
    if field_name not in ['fhr', 'up']:
        logger.warning(f"Denormalization only supported for 'fhr' and 'up' fields, got '{field_name}'. Returning data as-is.")
        return normalized_data
    
    stats = normalization_stats[field_name]
    
    if 'mean_tensor' in stats and 'std_tensor' in stats:
        mean_tensor = stats['mean_tensor'].to(device=normalized_data.device, dtype=normalized_data.dtype)
        std_tensor = stats['std_tensor'].to(device=normalized_data.device, dtype=normalized_data.dtype)
    else:
        mean_tensor = torch.tensor(stats['mean'], dtype=normalized_data.dtype, device=normalized_data.device)
        std_tensor = torch.tensor(stats['std'], dtype=normalized_data.dtype, device=normalized_data.device)    
    epsilon = 1e-8
    denormalized_data = normalized_data * (std_tensor + epsilon) + mean_tensor
    
    return denormalized_data
