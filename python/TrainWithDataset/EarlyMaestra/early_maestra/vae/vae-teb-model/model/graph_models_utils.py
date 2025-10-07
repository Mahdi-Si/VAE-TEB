import contextlib
import torch
from collections import OrderedDict
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
            "seqvae_model_state_dict",
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
