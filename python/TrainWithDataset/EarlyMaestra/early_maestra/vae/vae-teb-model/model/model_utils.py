from torch import nn
import torch
from typing import Tuple, Optional, Union, Dict, Any, Sequence
try:
    from torch._dynamo.eval_frame import OptimizedModule as _TorchOptimizedModule
except Exception:
    _TorchOptimizedModule = tuple()

from utils.custom_logger import setup_logging
setup_logging(
    log_to_file=True,
    log_to_console=True,
    file_path="my_service.log",
    file_level="DEBUG",
    console_level="INFO",
    rotation="100 MB",
    retention="14 days",
    compression="zip",
    serialize=False,
    backtrace=True,
    diagnose=False,
)

from loguru import logger as log
import logging as std_logging

DEFAULT_COMPILE_ATTEMPTS: Tuple[Dict[str, Any], ...] = (
    {
        "mode": "max-autotune-no-cudagraphs",
        "fullgraph": False,
        "dynamic": True,
    },
    {
        "mode": "reduce-overhead",
        "fullgraph": False,
        "dynamic": True,
        "options": {"triton.cudagraphs": False},
    },
)

def is_compiled_module(module: nn.Module) -> bool:
    """Return True if `module` is already wrapped by torch.compile."""
    if module is None:
        return False
    if hasattr(module, "_orig_mod"):
        return True
    try:
        return isinstance(module, _TorchOptimizedModule) if _TorchOptimizedModule else False
    except TypeError:
        return False

def ensure_compiled_module(
    module: nn.Module,
    *,
    compile_flag: bool = True,
    module_name: str = "module",
    attempts: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[nn.Module, bool]:
    """Wrap `module` with torch.compile using the provided attempts."""
    if module is None:
        return module, False
    if not compile_flag:
        return module, is_compiled_module(module)
    if not hasattr(torch, "compile"):
        log.warning(f"[{module_name}] torch.compile unavailable in this PyTorch build; running in eager mode")
        return module, False
    if is_compiled_module(module):
        return module, True

    compile_attempts = tuple(attempts) if attempts is not None else DEFAULT_COMPILE_ATTEMPTS
    for opts in compile_attempts:
        try:
            compiled = torch.compile(module, **opts)
            setattr(compiled, "_compile_options", opts)
            # Expose non-forward helpers (torch.compile only guarantees forward).
            passthrough_attrs = (
                "compute_loss",
                "encode_only",
                "forecast",
                "forecast_full",
                "aggregate_forecasts_to_canvas",
                "compute_forecast_loss",
                "evaluate_forecast_batch",
                "measure_transfer_entropy",
            )
            for attr in passthrough_attrs:
                if hasattr(module, attr):
                    setattr(compiled, attr, getattr(module, attr))
            log.info(f"[{module_name}] Compiled with torch.compile options={opts}")
            return compiled, True
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[{module_name}] torch.compile failed with options={opts}: {exc}")

    log.warning(f"[{module_name}] Falling back to eager mode after all torch.compile attempts failed")
    return module, False
