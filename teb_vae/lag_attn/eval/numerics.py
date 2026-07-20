r"""The pinned numeric environment for an evaluation run: fp32, no TF32, seeded.

Two of these settings are correctness requirements rather than preferences.

**TF32 is the non-obvious one.** It is on by default for matmul on Ampere and later, and it
carries $10$ mantissa bits. ``MEMORY_AND_EFFICIENCY.md`` rules out bf16 for $K_t$ because the
per-step KL is a small difference of larger quantities and bf16 has $8$; the same argument
applies to TF32 with two bits more headroom, which is not enough to make it safe by
inspection. It is therefore disabled explicitly rather than assumed away.

**``cudnn.benchmark`` is the reproducibility one.** ``configs/default.yaml`` sets it ``true``
for training, where it is the right trade. It selects convolution algorithms by timing them
at the first call of each shape, so the algorithm chosen -- and with it the summation order,
and with it the last bits of the result -- depends on what else the machine was doing. A
rerun with the same config, checkpoint and seed must produce identical numbers, so it is off.

Nothing here is training state, so calling it twice is harmless and calling it on a CPU-only
machine is a no-op for the CUDA half.
"""
from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch

#: Autocast device types to disable. Both, not just the one that is available: a stale
#: autocast flag on either would silently reintroduce reduced precision inside any module that
#: runs there, and disabling an unavailable device type is free.
_AUTOCAST_DEVICE_TYPES = ("cpu", "cuda")


def configure_numerics(seed: int) -> Dict[str, Any]:
    r"""Pin the numeric environment and seed every generator the pipeline draws from.

    Args:
        seed: Seed applied to ``random``, ``numpy`` and ``torch`` (including all CUDA
            devices). Recorded in ``summary.json`` so a run is reproducible from its own
            output.

    Returns:
        The effective settings, for recording. Every value is read back from global state
        *after* it was set rather than echoed from the assignment, so the dict reports what is
        actually in force -- including on a build where one of these knobs does not exist.
    """
    torch.set_default_dtype(torch.float32)

    # 'highest' is the fp32 path proper. Setting this and the two allow_tf32 flags is
    # belt-and-braces: they are separate switches in torch and a future version could route
    # around either one alone.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    for device_type in _AUTOCAST_DEVICE_TYPES:
        torch.set_autocast_enabled(device_type, False)

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    return {
        "seed": int(seed),
        "default_dtype": str(torch.get_default_dtype()),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "autocast_enabled": {
            device_type: bool(torch.is_autocast_enabled(device_type))
            for device_type in _AUTOCAST_DEVICE_TYPES
        },
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
