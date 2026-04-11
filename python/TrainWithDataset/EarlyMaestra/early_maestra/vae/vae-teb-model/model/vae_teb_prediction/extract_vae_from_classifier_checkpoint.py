"""Extract core VAE (SeqVae) weights from a classifier checkpoint.

The classification checkpoint (from PlSeqVaeClassifier or PlTemporalClassifier)
contains the full VaeTebTimeSeriesClassifier / TemporalVaeClassifier, which
wraps a frozen SeqVae as ``vae_model``. This script extracts just the VAE
weights and saves them in the format expected by ``load_checkpoint_strict()``.

Usage::

    python extract_vae_from_classifier_checkpoint.py \\
        --input /path/to/classifier-model-epoch=42.ckpt \\
        --output /path/to/core-model-extracted.ckpt

The output checkpoint can be loaded with::

    from model.vae_teb_prediction.vae_teb_model_prediction import SeqVae
    from train.graph_models_utils import load_checkpoint_strict

    vae = SeqVae()
    load_checkpoint_strict(vae, checkpoint="/path/to/core-model-extracted.ckpt")
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from loguru import logger


# Prefixes that the Lightning/wrapper checkpoint may add before ``vae_model.``
_WRAPPER_PREFIXES = [
    "model._orig_mod.vae_model.",
    "model._orig_model.vae_model.",
    "_orig_model.vae_model.",
    "_orig_mod.vae_model.",
    "model.model.vae_model.",
    "model.vae_model.",
    "vae_model.",
]


def extract_vae_state_dict(checkpoint_path: str, map_location: str = "cpu") -> OrderedDict:
    """Load a classifier checkpoint and extract the VAE sub-model state dict.

    Args:
        checkpoint_path: Path to the classifier ``.ckpt`` file.
        map_location: Device to map tensors to.

    Returns:
        OrderedDict with VAE-only keys (``source_encoder.``,
        ``target_encoder.``, ``conditional_encoder.``, ``decoder.``, etc.).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If no VAE keys are found in the checkpoint.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path}")
    ckpt = torch.load(str(path), map_location=map_location, weights_only=False)

    # Extract raw state_dict from Lightning checkpoint structure
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise RuntimeError(
            f"Cannot find state_dict in checkpoint. Top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    logger.info(f"Checkpoint has {len(state_dict)} keys total")

    # Extract VAE keys by trying each known prefix
    vae_state = OrderedDict()

    for key, value in state_dict.items():
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                vae_state[clean_key] = value
                break

    if not vae_state:
        # Diagnostic: show what prefixes exist
        sample_keys = list(state_dict.keys())[:20]
        raise RuntimeError(
            f"No VAE keys found with any known prefix. "
            f"Sample keys from checkpoint:\n" +
            "\n".join(f"  {k}" for k in sample_keys)
        )

    # Filter out any non-VAE artifacts (e.g. class_weights buffer)
    vae_expected_prefixes = (
        "source_encoder.", "target_encoder.", "conditional_encoder.",
        "decoder.", "forecaster.",
    )
    filtered = OrderedDict()
    skipped = []
    for key, value in vae_state.items():
        if any(key.startswith(p) for p in vae_expected_prefixes):
            filtered[key] = value
        else:
            skipped.append(key)

    if skipped:
        logger.info(f"Skipped {len(skipped)} non-encoder/decoder keys: {skipped[:5]}...")

    logger.info(f"Extracted {len(filtered)} VAE keys")
    return filtered


def save_as_core_checkpoint(vae_state: OrderedDict, output_path: str) -> None:
    """Save VAE state dict in the format expected by load_checkpoint_strict.

    Args:
        vae_state: OrderedDict of VAE parameters.
        output_path: Path to save the ``.ckpt`` file.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save as a simple dict with state_dict key (Lightning-compatible)
    torch.save({"state_dict": vae_state}, str(output))
    logger.info(f"Saved core VAE checkpoint to {output}")

    # Verify by loading back
    verify = torch.load(str(output), map_location="cpu", weights_only=False)
    n_keys = len(verify["state_dict"])
    total_params = sum(v.numel() for v in verify["state_dict"].values())
    logger.info(f"Verification: {n_keys} keys, {total_params:,} parameters")


def run(input_path: str, output_path: str, verify: bool = True) -> None:
    """Extract VAE from classifier checkpoint and save.

    Args:
        input_path: Path to classifier checkpoint (.ckpt).
        output_path: Path to save the extracted VAE checkpoint (.ckpt).
        verify: If True, verify by loading into a fresh SeqVae instance.
    """
    vae_state = extract_vae_state_dict(input_path)
    save_as_core_checkpoint(vae_state, output_path)

    if verify:
        logger.info("Verifying by loading into SeqVae...")
        try:
            from model.vae_teb_prediction.model.vae_teb_model_prediction import SeqVae
            from train.graph_models_utils import load_checkpoint_strict

            vae = SeqVae()
            result = load_checkpoint_strict(vae, checkpoint=output_path)
            if result is not None:
                logger.info("Verification PASSED: checkpoint loads cleanly into SeqVae")
            else:
                logger.error("Verification FAILED: load_checkpoint_strict returned None")
                sys.exit(1)
        except ImportError:
            logger.warning(
                "Could not import SeqVae (run from project root with proper PYTHONPATH). "
                "Skipping load verification."
            )

    logger.info("Done.")


def main():
    """CLI entry point using argparse."""
    parser = argparse.ArgumentParser(
        description="Extract core VAE checkpoint from a classifier checkpoint.",
    )
    parser.add_argument("--input", "-i", required=True, help="Classifier checkpoint path")
    parser.add_argument("--output", "-o", required=True, help="Output VAE checkpoint path")
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args()
    run(args.input, args.output, verify=args.verify)


if __name__ == "__main__":
    # ── Option 1: Set paths directly here ──────────────────────────────
    INPUT_CKPT = None   # e.g. "/path/to/classifier-model-epoch=42.ckpt"
    OUTPUT_CKPT = None  # e.g. "/path/to/core-model-extracted.ckpt"

    if INPUT_CKPT and OUTPUT_CKPT:
        run(INPUT_CKPT, OUTPUT_CKPT, verify=True)
    else:
        # ── Option 2: Fall back to CLI args ────────────────────────────
        main()
