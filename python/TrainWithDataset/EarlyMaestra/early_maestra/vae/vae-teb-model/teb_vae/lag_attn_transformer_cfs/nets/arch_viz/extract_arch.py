r"""Extract the ground-truth architecture of ``SeqVaeLagAttnTrfCfs`` into ``arch.json``.

The page ``arch_viz.html`` beside this file renders nothing it did not read from that JSON, so
everything the JSON states has to be *measured* off a constructed model and one real forward
pass rather than transcribed from the source. This module does three things:

1. **Builds the production model.** ``configs/default.yaml`` is swept onto the constructor
   exactly as the trainer sweeps it -- every ``model_config.VAE_model`` key naming a real
   constructor argument is forwarded -- and the causal warm-up budget
   (``causal_warmup_budget_steps``) and the channel alignment (``causal_align_reference``) are
   resolved into the four channel tuples and the two shift vectors through
   :func:`~teb_vae.lag_attn_cfs.causal_warmup.resolve_warmup_budget`. The production shards live
   on the training box, so the resolver reads the committed causal fixture shard, which carries
   the same per-channel warm-up and group-delay attributes. Every channel figure the page shows is
   read off the constructed model rather than named here, so a moved budget or reference re-costs
   the page instead of contradicting this paragraph.

2. **Traces one forward pass, three ways at once.**

   * :class:`torch.overrides.TorchFunctionMode` records **every torch call the forward makes at
     the level the source is written at** -- ``F.linear``, ``torch.cat``, ``x.transpose(1, 2)``
     -- with the real tensors that flowed through it, so every node has real shapes and dtypes.
     The mode is inactive inside its own handler, so a composite such as ``F.layer_norm`` is one
     node, not its decomposition.
   * ``sys.setprofile`` watches Python frames, so each call is attributed to the exact
     ``nn.Module`` *invocation* it ran inside, whether that invocation went through
     ``Module.__call__`` (``target_encoder``) or through a plain method (``horizon_core.decode``,
     ``lag_attn.build_lag_mask``), and to the repository function it ran in
     (``_reparameterize_shared``, ``entmax15``). Forward hooks are registered too, only to
     cross-check the frame-derived call counts.
   * Tensor **identity** links producers to consumers. Every output tensor is keyed in a
     :class:`torch.utils.weak.WeakTensorKeyDictionary`; when a later call consumes it, an edge is
     recorded. Views, in-place calls and pass-throughs (eval-mode dropout returns its input) are
     handled by re-keying the same object to the new node.

   ``torch.fx.symbolic_trace`` and ``torch.export`` are both attempted first and their failures
   recorded in the JSON: the forward reads ``int(y_st.shape[0])`` (FX sees a ``Proxy``) and
   branches on ``bool(tensor.any())`` (export cannot guard a data-dependent expression), so
   neither can trace this model, and the frame-plus-function-mode trace above is what ships.

3. **Measures what can be measured and labels what is estimated.** Parameter and buffer counts
   are read off the module tree; matmul-, convolution- and attention-class FLOPs are measured per
   module with :class:`torch.utils.flop_counter.FlopCounterMode`; per-call FLOPs are estimated
   from the recorded shapes and carry a ``flops_kind`` saying which formula produced them.
   Constructor arguments are captured by wrapping every ``nn.Module`` subclass ``__init__``
   before construction, along with the repository line that constructed each module.

**Reusing this for another model:** everything model-specific sits in one marked section near
the top (``MODEL_CLASS``, ``build_model_kwargs``, ``config_excerpt``, ``build_inputs``,
``model_geometry``); the rest reads only the model object. ``README.md`` beside this file walks
through it.

Run from the IDE's Run button (``RUN_ARGS`` at the bottom names the config, the shard, the batch
size and the output path) or from the command line::

    .venv/Scripts/python.exe -m teb_vae.lag_attn_transformer_cfs.nets.arch_viz.extract_arch \
        --batch-size 8
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import io
import json
import linecache
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Repository root: ``teb_vae/lag_attn_transformer_cfs/nets/arch_viz/extract_arch.py`` -> up five.
_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_THIS_FILE))))
)
# An IDE's Run button executes this file as a script, which puts *this directory* on sys.path
# rather than the repository root -- so the ``teb_vae.`` imports below would fail before
# ``__main__`` is reached. ``python -m`` from the repo root sets ``__package__`` and needs none of
# this, which is why the insert is guarded rather than unconditional.
if not __package__ and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402
from torch.utils._pytree import tree_flatten  # noqa: E402
from torch.utils.flop_counter import (  # noqa: E402
    FlopCounterMode,
    register_flop_formula,
    sdpa_flop_count,
)
from torch.utils.weak import WeakTensorKeyDictionary  # noqa: E402

from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_cfs.causal_warmup import resolve_warmup_budget  # noqa: E402
from teb_vae.lag_attn_cfs.model_kwargs import warmup_model_kwargs  # noqa: E402
from teb_vae.lag_attn_rws.eval.launch import resolve_launch_args  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.nets.model import SeqVaeLagAttnTrfCfs  # noqa: E402

#: Where the torch installation lives; frames under it are library frames, never call sites.
_TORCH_ROOT = os.path.dirname(os.path.abspath(torch.__file__))

#: Op families the page colours by. Every recorded call maps to exactly one; ``other`` is the
#: explicit fallback so an op the table does not know is rendered as such rather than mis-coloured.
_OP_FAMILY: Dict[str, str] = {
    # linear-algebra
    "linear": "linear",
    "matmul": "matmul", "mm": "matmul", "bmm": "matmul", "addmm": "matmul", "einsum": "matmul",
    "baddbmm": "matmul",
    # convolution
    "conv1d": "conv", "conv2d": "conv", "convolution": "conv",
    # normalisation
    "layer_norm": "norm", "group_norm": "norm", "batch_norm": "norm", "rms_norm": "norm",
    # attention kernels and their normalisers
    "scaled_dot_product_attention": "attention", "softmax": "attention",
    "log_softmax": "attention",
    # activations
    "gelu": "activation", "silu": "activation", "relu": "activation", "sigmoid": "activation",
    "tanh": "activation", "softplus": "activation", "elu": "activation", "mish": "activation",
    # regularisation
    "dropout": "dropout",
    # reductions and scans
    "mean": "reduce", "sum": "reduce", "max": "reduce", "min": "reduce", "any": "reduce",
    "all": "reduce", "cumsum": "reduce", "prod": "reduce", "sort": "reduce", "argmax": "reduce",
    "argsort": "reduce", "topk": "reduce", "std": "reduce", "var": "reduce", "norm": "reduce",
    # data movement: views, gathers, concatenations
    "view": "reshape", "reshape": "reshape", "transpose": "reshape", "permute": "reshape",
    "unsqueeze": "reshape", "squeeze": "reshape", "expand": "reshape", "flip": "reshape",
    "chunk": "reshape", "split": "reshape", "cat": "reshape", "stack": "reshape",
    "pad": "reshape", "unfold": "reshape", "contiguous": "reshape", "getitem": "reshape",
    "gather": "reshape", "index_select": "reshape", "flatten": "reshape", "clone": "reshape",
    "to": "reshape", "float": "reshape", "long": "reshape", "bool": "reshape",
    "type_as": "reshape", "narrow": "reshape", "select": "reshape", "repeat": "reshape",
    "masked_select": "reshape", "roll": "reshape", "movedim": "reshape", "t": "reshape",
    # tensor factories and randomness
    "arange": "generator", "randn_like": "generator", "randn": "generator", "zeros": "generator",
    "ones": "generator", "full": "generator", "zeros_like": "generator", "ones_like": "generator",
    "full_like": "generator", "tensor": "generator", "empty": "generator", "rand": "generator",
    "linspace": "generator", "eye": "generator",
}
#: Everything else that is a pure per-element map lands here by rule rather than by table.
_ELEMENTWISE = {
    "add", "sub", "mul", "div", "pow", "rsqrt", "sqrt", "exp", "log", "abs", "neg", "clamp",
    "clamp_min", "clamp_max", "where", "masked_fill", "nan_to_num", "ge", "gt", "le", "lt", "eq",
    "ne", "or", "and", "invert", "floordiv", "truediv", "rsub", "radd", "rmul", "rtruediv",
    "rpow", "logical_and", "logical_or", "logical_not", "sign", "square", "reciprocal", "floor",
    "ceil", "round", "cos", "sin", "erf", "log1p", "expm1", "lerp", "addcmul", "addcdiv",
    "maximum", "minimum", "fmod", "remainder", "xor", "bitwise_and", "bitwise_or", "isnan",
    "isinf", "logaddexp", "sigmoid_", "add_", "mul_", "sub_", "div_", "clamp_", "masked_fill_",
    "copy_", "fill_", "zero_", "index_fill_", "scatter_", "scatter_add_",
}

#: Python-operator spellings normalised to the op they perform, so ``x * y`` and ``torch.mul``
#: are one node kind. Reversed operators are the same op with swapped operands.
_OPERATOR_ALIASES = {
    "radd": "add", "rmul": "mul", "rsub": "sub", "rtruediv": "div", "truediv": "div",
    "rpow": "pow", "floordiv": "floordiv", "matmul": "matmul", "rmatmul": "matmul",
    "or": "or", "and": "and", "invert": "invert", "xor": "xor",
}


# ---------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------
def _relpath(filename: str) -> str:
    """Return ``filename`` relative to the repository root with forward slashes.

    Args:
        filename: An absolute path.

    Returns:
        The repo-relative path, or the absolute path when it is not under the repo.
    """
    try:
        rel = os.path.relpath(filename, _REPO_ROOT)
    except ValueError:  # different drive on Windows
        return filename.replace("\\", "/")
    if rel.startswith(".."):
        return filename.replace("\\", "/")
    return rel.replace("\\", "/")


def _is_repo_file(filename: str) -> bool:
    """Whether ``filename`` is repository code (not torch, not site-packages, not this tool)."""
    if not filename or filename.startswith("<"):
        return False
    absolute = os.path.abspath(filename)
    if absolute == _THIS_FILE:
        return False
    if not absolute.startswith(_REPO_ROOT):
        return False
    return ("site-packages" not in absolute) and (".venv" not in absolute)


def _is_torch_file(filename: str) -> bool:
    """Whether ``filename`` belongs to the torch installation."""
    return bool(filename) and os.path.abspath(filename).startswith(_TORCH_ROOT)


#: Every distinct call site seen, keyed ``"file:line"``, with a few lines of context. Filled by
#: :func:`_site` and written once into the JSON so a thousand ops do not carry a thousand copies.
_SITES: Dict[str, Dict[str, Any]] = {}


def _site(frame) -> Dict[str, Any]:
    """Describe one frame as a call site: file, line, function and the source line itself.

    Args:
        frame: A Python frame object.

    Returns:
        ``{"file", "line", "function", "code", "key"}`` with the file repo-relative; the key
        indexes the shared ``sites`` table, which also holds two lines of context either side.
    """
    filename = frame.f_code.co_filename
    line = int(frame.f_lineno)
    code = linecache.getline(filename, line).strip()
    key = f"{_relpath(filename)}:{line}"
    if key not in _SITES:
        context = [
            {"line": n, "code": linecache.getline(filename, n).rstrip("\n")}
            for n in range(max(1, line - 2), line + 3)
            if linecache.getline(filename, n)
        ]
        _SITES[key] = {
            "file": _relpath(filename),
            "line": line,
            "function": frame.f_code.co_qualname,
            "context": context,
        }
    return {
        "file": _relpath(filename),
        "line": line,
        "function": frame.f_code.co_qualname,
        "code": code,
        "key": key,
    }


def _repo_call_chain(start_depth: int = 1, limit: int = 6) -> List[Dict[str, Any]]:
    """Walk outward from the caller and collect the repository frames on the stack.

    Torch, site-packages and this tool's own frames are skipped, so the first entry is the
    repository line that actually issued the call -- ``self.proj_in(self.norm_in(x))`` rather than
    ``torch/nn/modules/linear.py``.

    Args:
        start_depth: How many frames above this helper to start from.
        limit: Maximum number of repository frames to return, innermost first.

    Returns:
        A list of :func:`_site` dicts, innermost first; empty if no repository frame is on the
        stack.
    """
    chain: List[Dict[str, Any]] = []
    frame = sys._getframe(start_depth)
    while frame is not None and len(chain) < limit:
        if _is_repo_file(frame.f_code.co_filename):
            chain.append(_site(frame))
        frame = frame.f_back
    return chain


def _library_function(start_depth: int = 1) -> Optional[str]:
    """Name the innermost *third-party* Python function on the stack, if any (e.g. ``entmax15``).

    Torch's own frames do not count -- ``F.layer_norm`` is the op itself, not a library the op ran
    inside -- but a pure-Python library such as ``entmax`` does, because its internals appear as
    many small ops that a reader wants grouped under the function that issued them.

    Args:
        start_depth: How many frames above this helper to start from.

    Returns:
        ``"module.qualname"`` of the innermost such frame, or ``None``.
    """
    frame = sys._getframe(start_depth)
    while frame is not None:
        filename = frame.f_code.co_filename
        if (
            filename
            and not filename.startswith("<")
            and not _is_torch_file(filename)
            and not _is_repo_file(filename)
            and os.path.abspath(filename) != _THIS_FILE
            and "site-packages" in os.path.abspath(filename)
        ):
            module_name = frame.f_globals.get("__name__", "?")
            return f"{module_name}.{frame.f_code.co_qualname}"
        frame = frame.f_back
    return None


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Turn a constructor argument or op argument into something ``json.dumps`` accepts.

    Args:
        value: Any Python object.
        depth: Recursion depth guard.

    Returns:
        A JSON-serialisable rendering. Modules become ``"<ClassName>"``, tensors become their
        shape, classes their qualified name, everything unknown its ``repr``.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return repr(value)
        return value
    if depth > 4:
        return repr(value)[:120]
    if isinstance(value, nn.Module):
        return f"<{type(value).__name__}>"
    if isinstance(value, torch.Tensor):
        return {"tensor": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        return getattr(value, "__qualname__", None) or getattr(value, "__name__", repr(value))
    return repr(value)[:120]


def _index_repr(index: Any) -> str:
    """Render a ``__getitem__`` index the way it was written: ``[:, :, 0, :]``.

    Args:
        index: The index object passed to ``Tensor.__getitem__``.

    Returns:
        A short string.
    """
    def one(item: Any) -> str:
        if item is Ellipsis:
            return "..."
        if item is None:
            return "None"
        if isinstance(item, slice):
            start = "" if item.start is None else str(item.start)
            stop = "" if item.stop is None else str(item.stop)
            if item.step is None:
                return f"{start}:{stop}"
            return f"{start}:{stop}:{item.step}"
        if isinstance(item, torch.Tensor):
            return f"tensor{list(item.shape)}"
        return repr(item)

    if isinstance(index, tuple):
        return "[" + ", ".join(one(item) for item in index) + "]"
    return "[" + one(index) + "]"


def _op_name(func: Any) -> str:
    """Normalise a torch-function callable to a short op name.

    ``TensorBase.mul`` -> ``mul``, ``_VariableFunctionsClass.cat`` -> ``cat``,
    ``Tensor.__rsub__`` -> ``sub``, ``TensorBase.__getitem__`` -> ``getitem``.

    Args:
        func: The callable ``__torch_function__`` received.

    Returns:
        The short name.
    """
    name = getattr(func, "__name__", None) or getattr(func, "__qualname__", None) or str(func)
    name = name.split(".")[-1]
    stripped = name.strip("_")
    if stripped in _OPERATOR_ALIASES:
        return _OPERATOR_ALIASES[stripped]
    return stripped or name


def _family_of(op: str) -> str:
    """Map a normalised op name to its colour family."""
    if op in _OP_FAMILY:
        return _OP_FAMILY[op]
    if op in _ELEMENTWISE:
        return "elementwise"
    return "other"


def _module_family(module: nn.Module) -> str:
    """Map a module to the family the page colours it with, by class."""
    name = type(module).__name__
    if isinstance(module, (nn.Linear,)):
        return "linear"
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return "conv"
    if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d)) or "Norm" in name:
        return "norm"
    if "Attention" in name:
        return "attention"
    if isinstance(module, (nn.GELU, nn.SiLU, nn.ReLU, nn.Sigmoid, nn.Tanh)):
        return "activation"
    if isinstance(module, nn.Dropout):
        return "dropout"
    if name == "LayerScale":
        return "elementwise"
    if isinstance(module, nn.Identity):
        return "reshape"
    if name in ("ChannelGate", "ChannelDelay"):
        return "reshape"
    if name in ("RotaryPositionEncoding",):
        return "elementwise"
    if name in ("TEAnalysisHead",):
        return "reduce"
    if name in ("SwiGLUFeedForward", "ResidualMLP", "GatedCausalConvBlock"):
        return "mlp"
    return "container"


def _shape_of(tensor: torch.Tensor) -> List[int]:
    """List form of a tensor's shape."""
    return [int(dim) for dim in tensor.shape]


def _dtype_of(tensor: torch.Tensor) -> str:
    """Short dtype string: ``float32``, ``int64``, ``bool``."""
    return str(tensor.dtype).replace("torch.", "")


def _prod(values: Sequence[int]) -> int:
    """Product of a sequence of ints; ``1`` for the empty sequence."""
    result = 1
    for value in values:
        result *= int(value)
    return result


# ---------------------------------------------------------------------------------------
# FLOP estimation per recorded call
# ---------------------------------------------------------------------------------------
def _estimate_flops(
    op: str,
    in_shapes: List[List[int]],
    out_shapes: List[List[int]],
    positional: List[Any],
) -> Tuple[Optional[int], str]:
    r"""Estimate the floating-point operations of one recorded call from its shapes.

    Matmul-class formulas match :mod:`torch.utils.flop_counter` (which is what the module-level
    ``flops_measured`` field is), so the two can be cross-checked; everything else is a stated
    convention -- one FLOP per output element for a per-element map, $5N$ for a normalisation,
    zero for pure data movement.

    Args:
        op: Normalised op name.
        in_shapes: Shapes of the tensor inputs, in argument order (parameters included).
        out_shapes: Shapes of the tensor outputs.
        positional: The non-tensor positional arguments (the einsum equation lives here).

    Returns:
        ``(flops, kind)`` where ``kind`` names the formula: ``matmul`` (comparable to the
        measured counter), ``elementwise`` (estimate), ``norm`` (estimate), ``movement`` (zero
        by convention), ``none`` (unknown, ``flops`` is ``None``).
    """
    numel_out = sum(_prod(shape) for shape in out_shapes) if out_shapes else 0
    numel_in0 = _prod(in_shapes[0]) if in_shapes else 0
    try:
        if op == "linear" and len(in_shapes) >= 2:
            x, w = in_shapes[0], in_shapes[1]
            return 2 * _prod(x[:-1]) * w[1] * w[0], "matmul"
        if op in ("conv1d", "convolution") and len(in_shapes) >= 2 and out_shapes:
            w, out = in_shapes[1], out_shapes[0]
            return 2 * _prod(out) * _prod(w[1:]), "matmul"
        if op == "einsum" and positional and isinstance(positional[0], str):
            equation = positional[0].replace(" ", "")
            lhs = equation.split("->")[0].split(",")
            sizes: Dict[str, int] = {}
            for term, shape in zip(lhs, in_shapes):
                for letter, dim in zip(term, shape):
                    sizes[letter] = dim
            return 2 * _prod(list(sizes.values())), "matmul"
        if op in ("matmul", "mm", "bmm", "addmm", "baddbmm") and len(in_shapes) >= 2 and out_shapes:
            a = in_shapes[-2] if op in ("addmm", "baddbmm") else in_shapes[0]
            return 2 * _prod(out_shapes[0]) * a[-1], "matmul"
        if op == "scaled_dot_product_attention" and len(in_shapes) >= 3:
            q, k, v = in_shapes[0], in_shapes[1], in_shapes[2]
            batch = _prod(q[:-2])
            return 2 * batch * q[-2] * k[-2] * q[-1] + 2 * batch * q[-2] * k[-2] * v[-1], "matmul"
        if op in ("layer_norm", "group_norm", "batch_norm", "rms_norm"):
            return 5 * numel_in0, "norm"
        if op in ("softmax", "log_softmax"):
            return 5 * numel_in0, "elementwise"
        if op == "sort" and in_shapes:
            last = max(1, in_shapes[0][-1]) if in_shapes[0] else 1
            return int(numel_in0 * max(1.0, math.log2(last))), "elementwise"
        if _family_of(op) in ("activation", "elementwise"):
            return numel_out, "elementwise"
        if _family_of(op) == "reduce":
            return numel_in0, "elementwise"
        if _family_of(op) in ("reshape", "generator", "dropout"):
            return 0, "movement"
    except (IndexError, TypeError, ValueError):
        return None, "none"
    return None, "none"


def _register_cpu_sdpa_flops() -> None:
    """Teach :mod:`torch.utils.flop_counter` about the CPU flash-attention kernel.

    torch 2.7 registers formulas for the CUDA/cuDNN/efficient attention kernels but not for
    ``aten._scaled_dot_product_flash_attention_for_cpu``, so on a CPU trace every
    ``F.scaled_dot_product_attention`` call is silently *uncounted* -- which is exactly the kind
    of gap this tool exists to close. Same formula as torch's own ``sdpa_flop``.
    """
    op = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu", None)
    if op is None:
        return

    def cpu_sdpa(query, key, value, *args, out_val=None, **kwargs):
        return sdpa_flop_count(query.shape, key.shape, value.shape)

    register_flop_formula(op, get_raw=True)(cpu_sdpa)


# ---------------------------------------------------------------------------------------
# Constructor-argument capture
# ---------------------------------------------------------------------------------------
def _wrap_init(cls: type) -> None:
    """Wrap ``cls.__init__`` so every instance records the arguments it was constructed with.

    The most-derived class's wrapper runs first, so ``setdefault`` semantics keep *its* arguments
    and ignore the ``super().__init__`` calls beneath. Written into ``__dict__`` directly because
    ``nn.Module.__setattr__`` is not usable before ``nn.Module.__init__`` has run.

    Args:
        cls: An ``nn.Module`` subclass that defines its own ``__init__``.
    """
    original = cls.__dict__.get("__init__")
    if original is None or getattr(original, "_arch_wrapped", False):
        return
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if "_arch_ctor" not in self.__dict__:
            record: Dict[str, Any]
            try:
                bound = signature.bind(self, *args, **kwargs)
                explicit = {k: _jsonable(v) for k, v in bound.arguments.items() if k != "self"}
                bound.apply_defaults()
                full = {k: _jsonable(v) for k, v in bound.arguments.items() if k != "self"}
                record = {"explicit": explicit, "all": full}
            except TypeError:
                record = {
                    "explicit": {"*args": _jsonable(args), "**kwargs": _jsonable(kwargs)},
                    "all": {},
                }
            record["class"] = cls.__qualname__
            chain = _repo_call_chain(start_depth=2, limit=3)
            record["site"] = chain[0] if chain else None
            object.__setattr__(self, "_arch_ctor", record)
        return original(self, *args, **kwargs)

    wrapped._arch_wrapped = True  # type: ignore[attr-defined]
    cls.__init__ = wrapped  # type: ignore[assignment]


def install_ctor_capture() -> int:
    """Wrap ``__init__`` on every ``nn.Module`` subclass torch and ``teb_vae`` have loaded.

    Must run *after* the model module is imported (so its classes exist) and *before* the model
    is constructed. Returns the number of classes wrapped so the caller can assert it was not a
    silent no-op.

    Returns:
        The count of wrapped classes.
    """
    classes: List[type] = []
    for module_name, module in list(sys.modules.items()):
        if module is None:
            continue
        if not (module_name.startswith("teb_vae") or module_name.startswith("torch.nn.modules")):
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, nn.Module) and obj is not nn.Module and "__init__" in obj.__dict__:
                classes.append(obj)
    for cls in set(classes):
        _wrap_init(cls)
    return len(set(classes))


# =======================================================================================
# MODEL-SPECIFIC SECTION -- the only code to edit for another model (see README.md).
#
# Five hooks: which class, how its constructor kwargs are obtained, what a realistic dummy input
# is, which configuration excerpt to record, and which geometry numbers the overview tabulates.
# Everything outside this section is model-agnostic: it reads the model object, calls it with
# whatever ``build_inputs`` returned, and never names an attribute of this architecture.
# =======================================================================================
MODEL_CLASS = SeqVaeLagAttnTrfCfs

#: Repo-root-relative defaults for ``--config`` and ``--shard``.
DEFAULT_CONFIG = "teb_vae/lag_attn_transformer_cfs/configs/default.yaml"
DEFAULT_SHARD = "teb_vae/lag_attn/tests/fixtures/tiny_shard_causal.hdf5"

#: The one ``VAE_model`` key whose YAML ``null`` is a *value* for this architecture rather than
#: "use the constructor default": an unbounded source encoder is ``source_attention_window: null``.
#: Mirrors ``teb_vae.lag_attn_transformer_rws.trainer.NULLABLE_MODEL_KEYS`` without importing the
#: trainer, which would pull Lightning into a tool that only needs the net.
_NULLABLE_MODEL_KEYS = ("source_attention_window",)

#: Config keys the trainer never forwards to the constructor even though the name matches.
_NON_CONSTRUCTOR_KEYS = frozenset({"init_weights"})


def build_model_kwargs(config_path: str, shard_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Sweep the YAML onto the constructor and resolve the warm-up budget, as the trainer does.

    For a model without a YAML this hook can simply return ``(dict(...), {}, "hand-written kwargs")``.

    Args:
        config_path: The YAML config; its ``base:`` chain is resolved through the shared loader.
        shard_path: The causal HDF5 shard the warm-up resolver reads the per-channel warm-up
            attributes from. Substituted for both dataset splits.

    Returns:
        ``(model_kwargs, config, one-line summary of how the kwargs were resolved)``.
    """
    config = load_config(config_path)
    dataset = config.setdefault("dataset_config", {})
    dataset["vae_train_datasets"] = [shard_path]
    dataset["vae_test_datasets"] = [shard_path]

    vae_config = (config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
    valid = set(inspect.signature(MODEL_CLASS.__init__).parameters)
    kwargs = {
        name: value
        for name, value in vae_config.items()
        if name in valid and name not in _NON_CONSTRUCTOR_KEYS and value is not None
    }
    # Re-admit the one key whose ``null`` is a value (see _NULLABLE_MODEL_KEYS).
    for name in _NULLABLE_MODEL_KEYS:
        if name in vae_config and vae_config[name] is None:
            kwargs[name] = None

    budget = resolve_warmup_budget(config)
    summary = "no causal warm-up budget configured (ungated model)"
    if budget is not None:
        kwargs.update(warmup_model_kwargs(budget, MODEL_CLASS))
        summary = budget.summary()
    return kwargs, config, summary


def config_excerpt(config: Dict[str, Any]) -> Dict[str, Any]:
    """The part of the configuration worth recording beside the constructor kwargs.

    Args:
        config: The resolved configuration ``build_model_kwargs`` returned.

    Returns:
        A JSON-serialisable mapping; ``{}`` when there is no configuration.
    """
    vae = (config.get("model_config", {}) or {}).get("VAE_model", {}) or {}
    return {k: _jsonable(v) for k, v in vae.items()}


def build_inputs(
    model: nn.Module, batch_size: int, seed: int
) -> Tuple[Dict[str, torch.Tensor], Tuple[str, ...], Tuple[str, ...], Dict[str, str]]:
    """Realistic dummy inputs for one forward pass, at the widths the model was built for.

    The loader z-scores every feature stream, so unit-normal draws are the realistic scale; the
    shipped tiling stride needs a per-sample anchor phase in $[0, S)$.

    Args:
        model: The constructed model (read for its declared widths and stride).
        batch_size: $B$.
        seed: Seed for the draws.

    Returns:
        ``(inputs, positional, keyword, meanings)``: the named tensors in call order, which names
        are passed positionally and which as keywords, and a one-line meaning per input.
    """
    seq_len = int(model.sequence_length)
    st_width = int(model.c_y) - 66  # fhr_st | fhr_ph; the phase-harmonic block is 66 wide
    torch.manual_seed(seed)
    inputs = {
        "y_st": torch.randn(batch_size, seq_len, st_width),
        "y_ph": torch.randn(batch_size, seq_len, 66),
        "u_stream": torch.randn(batch_size, seq_len, int(model.c_u)),
        "anchor_phase": torch.arange(batch_size) % int(model.anchor_stride),
    }
    meanings = {
        "y_st": "target scattering coefficients (fhr_st), z-scored by the loader",
        "y_ph": "target phase-harmonic coefficients (fhr_ph), z-scored",
        "u_stream": "source stream up_st | up_ph (36 + 15 channels), z-scored",
        "anchor_phase": "per-sample tile phase in [0, anchor_stride)",
    }
    return inputs, ("y_st", "y_ph", "u_stream"), ("anchor_phase",), meanings


def model_geometry(model: nn.Module) -> Dict[str, Any]:
    """The geometry numbers the page's overview tabulates, read off the constructed model.

    Args:
        model: The constructed model.

    Returns:
        A flat, JSON-serialisable mapping; the page renders whatever keys it finds.
    """
    return {
        "sequence_length": int(model.sequence_length),
        "horizon": int(model.horizon),
        "raw_per_step": int(model.raw_per_step),
        "warmup_period": int(model.warmup_period),
        "t_valid": int(model.geometry.t_valid),
        "anchor_stride": int(model.anchor_stride),
        "lag_floor": int(model.lag_floor),
        "max_lag": int(model.max_lag),
        "L": int(model.lag_attn.L),
        "c_y": int(model.c_y),
        "c_u": int(model.c_u),
        "c_keep_target": int(model.target_gate.out_channels) if model.target_gate is not None else int(model.c_y),
        "c_keep_source": int(model.source_gate.out_channels) if model.source_gate is not None else int(model.c_u),
        "decoder_out_channels": int(model.decoder_out_channels),
        "d_model": int(model.d_model),
        "d_z": int(model.d_z),
        "num_heads": int(model.num_heads),
        "target_encoder_receptive_field": model.target_encoder.receptive_field,
        # Read off whichever module `lag_kv_source` put on the source side, because that is what
        # the lag attention scores. Under a local arm there is no source encoder to read at all,
        # and the adapter alone reaches exactly one step.
        "lag_kv_source": model.lag_kv_source,
        "source_encoder_receptive_field": (
            1 if model.source_kv_body() is None else model.source_kv_body().receptive_field
        ),
        "conv_reach": int(model.target_encoder.conv_reach),
        "target_warmup_max": max(model.target_warmup_steps or (0,)),
        "source_warmup_max": max(model.source_warmup_steps or (0,)),
        "base_decode": model.base_decode,
        "posterior_logvar_mode": model.posterior_logvar_mode,
    }
# =======================================================================================
# End of the model-specific section.
# =======================================================================================


# ---------------------------------------------------------------------------------------
# The tracer
# ---------------------------------------------------------------------------------------
class ArchTracer(TorchFunctionMode):
    r"""Record every torch call of one forward pass, with scopes, shapes and dataflow edges.

    Three mechanisms cooperate (see the module docstring): the ``__torch_function__`` handler
    records calls; a ``sys.setprofile`` hook maintains the scope stack from Python frames; a weak
    tensor-keyed dictionary links producers to consumers by identity.

    Attributes:
        ops: The recorded calls, in execution order.
        tensors: Metadata per tensor id -- shape, dtype, bytes, producer, consumers.
        scopes: Module invocations and repository functions the calls ran inside.
    """

    def __init__(
        self,
        model: nn.Module,
        module_paths: Dict[int, str],
        param_names: Dict[int, str],
        buffer_names: Dict[int, str],
    ) -> None:
        """Initialise the tracer over a model whose module and parameter identities are known.

        Args:
            model: The model that will be run under this tracer.
            module_paths: ``id(module) -> primary dotted path`` for every module of the tree.
            param_names: ``id(parameter) -> dotted name``.
            buffer_names: ``id(buffer) -> dotted name``.
        """
        super().__init__()
        self.model = model
        self.module_paths = module_paths
        self.param_names = param_names
        self.buffer_names = buffer_names

        self.ops: List[Dict[str, Any]] = []
        self.tensors: Dict[str, Dict[str, Any]] = {}
        self.scopes: Dict[str, Dict[str, Any]] = {}
        self._tensor_ids: WeakTensorKeyDictionary = WeakTensorKeyDictionary()
        self._tensor_counter = 0
        self._scope_counter = 0
        self._scope_stack: List[str] = []
        self._frame_scope: Dict[int, str] = {}
        self._frame_module: Dict[int, int] = {}
        self._module_depth: Dict[int, int] = defaultdict(int)
        self._module_calls: Dict[int, int] = defaultdict(int)
        self.skipped_non_tensor_calls = 0
        self.untraced_inputs = 0

        self.root_scope = self._open_scope(
            kind="module", module_id=id(model), method="forward", frame=None
        )
        self._scope_stack.append(self.root_scope)
        # The root scope *is* the model's forward invocation, created here so it exists before
        # any frame is entered; marking the model as already one frame deep stops its own
        # ``_call_impl`` frame from opening a second scope for the same invocation.
        self._module_depth[id(model)] = 1
        self._module_forward_calls: Dict[int, int] = defaultdict(int)
        self._module_forward_calls[id(model)] = 1

    # ---- tensor bookkeeping ------------------------------------------------------------
    def _new_tensor(self, tensor: torch.Tensor, producer: Optional[str], role: str) -> str:
        """Register a tensor object under a fresh id and return that id."""
        self._tensor_counter += 1
        tid = f"t{self._tensor_counter}"
        self._tensor_ids[tensor] = tid
        self.tensors[tid] = {
            "id": tid,
            "shape": _shape_of(tensor),
            "dtype": _dtype_of(tensor),
            "numel": int(tensor.numel()),
            "bytes": int(tensor.numel() * tensor.element_size()),
            "producer": producer,
            "role": role,
            "consumers": [],
            "is_view": False,
        }
        return tid

    def register_input(self, name: str, tensor: torch.Tensor) -> str:
        """Register a model input so edges out of it are labelled by name.

        Args:
            name: The forward argument name.
            tensor: The tensor passed under that name.

        Returns:
            The tensor id.
        """
        tid = self._new_tensor(tensor, producer=None, role="input")
        self.tensors[tid]["name"] = name
        return tid

    def lookup(self, tensor: torch.Tensor) -> Optional[str]:
        """Return the id under which ``tensor`` is currently known, if any."""
        return self._tensor_ids.get(tensor)

    # ---- scope bookkeeping -------------------------------------------------------------
    def _open_scope(
        self,
        *,
        kind: str,
        frame,
        module_id: Optional[int] = None,
        method: Optional[str] = None,
        function: Optional[str] = None,
    ) -> str:
        """Create a scope record and return its id (the caller pushes it)."""
        self._scope_counter += 1
        sid = f"s{self._scope_counter}"
        parent = self._scope_stack[-1] if self._scope_stack else None
        record: Dict[str, Any] = {
            "id": sid,
            "kind": kind,
            "parent": parent,
            "children": [],
            "ops": [],
            "seq": len(self.ops),
        }
        if module_id is not None:
            self._module_calls[module_id] += 1
            record["module"] = self.module_paths[module_id]
            record["method"] = method
            record["call_index"] = self._module_calls[module_id] - 1
        if function is not None:
            record["function"] = function
        if frame is not None:
            record["site"] = _site(frame)
        if parent is not None:
            self.scopes[parent]["children"].append(sid)
        self.scopes[sid] = record
        return sid

    def _profile(self, frame, event: str, arg) -> None:
        """The ``sys.setprofile`` hook: open scopes on frame entry, close them on exit."""
        if event == "call":
            self._on_call(frame)
        elif event == "return":
            self._on_return(frame)

    def _on_call(self, frame) -> None:
        """Decide whether an entered frame opens a module or function scope."""
        code = frame.f_code
        filename = code.co_filename
        if filename == _THIS_FILE or filename.startswith("<"):
            return
        module_obj = None
        if code.co_argcount >= 1 and code.co_varnames[0] == "self":
            candidate = frame.f_locals.get("self")
            if candidate is not None and id(candidate) in self.module_paths:
                module_obj = candidate
        if module_obj is not None:
            mid = id(module_obj)
            depth = self._module_depth[mid]
            self._module_depth[mid] = depth + 1
            self._frame_module[id(frame)] = mid
            if depth == 0 and not code.co_name.startswith("__"):
                if code.co_name in ("_wrapped_call_impl", "_call_impl"):
                    # A genuine nn.Module.__call__: the module scope proper.
                    self._module_forward_calls[mid] += 1
                    sid = self._open_scope(
                        kind="module", module_id=mid, method="forward", frame=frame
                    )
                else:
                    # A method invoked directly (horizon_core.decode, lag_attn.build_lag_mask):
                    # a scope that carries the module for the reader but is a method call.
                    sid = self._open_scope(
                        kind="method", module_id=mid, method=code.co_name, frame=frame
                    )
                self._frame_scope[id(frame)] = sid
                self._scope_stack.append(sid)
                return
            # A nested method of an already-open module: only ``forward`` under ``_call_impl``
            # is the scope itself; any other repository method becomes a function scope below.
            if code.co_name == "forward":
                top = self.scopes[self._scope_stack[-1]]
                if top.get("module") == self.module_paths[mid] and top["kind"] == "module":
                    return
        # Repository (or third-party pure-Python) function frames become function scopes so
        # ``_reparameterize_shared``, ``smooth_bound`` and ``entmax15`` group their ops.
        if _is_repo_file(filename) or (
            not _is_torch_file(filename) and "site-packages" in os.path.abspath(filename)
        ):
            if code.co_name.startswith("__") or code.co_name.startswith("<"):
                return
            qualname = code.co_qualname
            if module_obj is not None:
                qualname = f"{type(module_obj).__name__}.{code.co_name}"
            sid = self._open_scope(kind="function", function=qualname, frame=frame)
            self._frame_scope[id(frame)] = sid
            self._scope_stack.append(sid)

    def _on_return(self, frame) -> None:
        """Close whatever the frame opened."""
        fid = id(frame)
        mid = self._frame_module.pop(fid, None)
        if mid is not None:
            self._module_depth[mid] -= 1
        sid = self._frame_scope.pop(fid, None)
        if sid is not None:
            # Frames nest, so the scope being closed is the top of the stack; anything above it
            # would be a frame that exited without a return event, which is closed with it.
            while self._scope_stack and self._scope_stack[-1] != sid:
                self._scope_stack.pop()
            if self._scope_stack:
                self._scope_stack.pop()

    def __enter__(self):
        sys.setprofile(self._profile)
        return super().__enter__()

    def __exit__(self, *exc):
        sys.setprofile(None)
        return super().__exit__(*exc)

    # ---- the handler ------------------------------------------------------------------
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        try:
            self._record(func, args, kwargs, out)
        except Exception:  # pragma: no cover - a recording failure must not break the forward
            traceback.print_exc()
        return out

    def _record(self, func, args, kwargs, out) -> None:
        """Record one call: inputs by identity, outputs under fresh ids, scope, site, FLOPs."""
        op = _op_name(func)
        if op in ("set_grad_enabled", "get", "getattribute"):
            return
        flat_out, _ = tree_flatten(out)
        out_tensors = [t for t in flat_out if isinstance(t, torch.Tensor)]
        if not out_tensors:
            self.skipped_non_tensor_calls += 1
            return

        # Inputs, in argument order: tensors resolve to ids / parameters / buffers, everything
        # else is summarised for the panel.
        flat_args, _ = tree_flatten(args)
        flat_kwargs, _ = tree_flatten(kwargs)
        inputs: List[Dict[str, Any]] = []
        in_shapes: List[List[int]] = []
        input_storage_ptrs = set()
        for tensor in [t for t in flat_args + flat_kwargs if isinstance(t, torch.Tensor)]:
            in_shapes.append(_shape_of(tensor))
            try:
                input_storage_ptrs.add(tensor.untyped_storage().data_ptr())
            except RuntimeError:
                pass
            pid = id(tensor)
            if pid in self.param_names:
                inputs.append({"kind": "param", "name": self.param_names[pid],
                               "shape": _shape_of(tensor)})
                continue
            if pid in self.buffer_names:
                inputs.append({"kind": "buffer", "name": self.buffer_names[pid],
                               "shape": _shape_of(tensor)})
                continue
            tid = self._tensor_ids.get(tensor)
            if tid is None:
                # A tensor no recorded call produced: rendered as an explicit "untraced" source.
                self.untraced_inputs += 1
                tid = self._new_tensor(tensor, producer=None, role="untraced")
            inputs.append({"kind": "tensor", "id": tid})

        positional = [
            _index_repr(a) if op == "getitem" and i == 1 else _jsonable(a)
            for i, a in enumerate(args)
            if not isinstance(a, torch.Tensor)
        ]
        # tree_flatten already emptied nested tensor lists; drop the leftover empty containers.
        positional = [p for p in positional if p not in ([], {})]
        keyword = {k: _jsonable(v) for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}

        seq = len(self.ops)
        oid = f"op{seq}"
        outputs: List[str] = []
        for tensor in out_tensors:
            try:
                same_storage = tensor.untyped_storage().data_ptr() in input_storage_ptrs
            except RuntimeError:
                same_storage = False
            is_view = bool(tensor._is_view()) or same_storage
            tid = self._new_tensor(tensor, producer=oid, role="activation")
            self.tensors[tid]["is_view"] = is_view
            outputs.append(tid)
        for entry in inputs:
            if entry["kind"] == "tensor":
                self.tensors[entry["id"]]["consumers"].append(oid)

        scope_id = self._scope_stack[-1]
        module_path = self._innermost_module(scope_id)
        chain = _repo_call_chain(start_depth=3, limit=5)
        flops, flops_kind = _estimate_flops(op, in_shapes, [self.tensors[t]["shape"] for t in outputs], positional)
        self.ops.append({
            "id": oid,
            "seq": seq,
            "op": op,
            "family": _family_of(op),
            "scope": scope_id,
            "module": module_path,
            "site": chain[0] if chain else None,
            "chain": chain,
            "library_fn": _library_function(start_depth=3),
            "inputs": inputs,
            "outputs": outputs,
            "args": positional,
            "kwargs": keyword,
            "flops": flops,
            "flops_kind": flops_kind,
        })
        self.scopes[scope_id]["ops"].append(oid)

    def _innermost_module(self, scope_id: str) -> str:
        """The dotted path of the nearest enclosing module or method scope."""
        sid: Optional[str] = scope_id
        while sid is not None:
            record = self.scopes[sid]
            if "module" in record:
                return record["module"]
            sid = record["parent"]
        return ""


# ---------------------------------------------------------------------------------------
# Static description of the module tree
# ---------------------------------------------------------------------------------------
def _parse_arg_docs(cls: type) -> Dict[str, Dict[str, Any]]:
    """Read the ``Args:`` section of ``cls.__init__``'s docstring, one entry per argument.

    Google-style: an ``Args:`` line, then ``name: text`` entries indented one level deeper, each
    continued by lines indented deeper still. Every entry records the source line it starts on,
    so the page can cite the sentence that explains a hyperparameter rather than paraphrase it.

    Args:
        cls: The class whose constructor docstring to read.

    Returns:
        ``{arg_name: {"line": int, "file": str, "text": str}}``; empty when there is no section.
    """
    init = cls.__dict__.get("__init__")
    if init is None:
        return {}
    init = getattr(init, "__wrapped__", init)  # the capture wrapper, if installed
    try:
        lines, start = inspect.getsourcelines(init)
        source_file = _relpath(inspect.getsourcefile(init) or "")
    except (OSError, TypeError):
        return {}
    docs: Dict[str, Dict[str, Any]] = {}
    args_indent: Optional[int] = None
    entry_indent: Optional[int] = None
    current: Optional[str] = None
    for offset, raw in enumerate(lines):
        line = raw.rstrip("\n")
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if args_indent is None:
            if stripped == "Args:":
                args_indent = indent
            continue
        if not stripped:
            continue
        if indent <= args_indent:
            break  # another section (Raises:, Returns:) or the docstring's end
        if entry_indent is None:
            entry_indent = indent
        head = stripped.split(":", 1)[0]
        if indent == entry_indent and ":" in stripped and head.isidentifier():
            current = head
            docs[current] = {"line": start + offset, "file": source_file,
                             "text": stripped.split(":", 1)[1].strip()}
        elif current is not None and indent > entry_indent:
            docs[current]["text"] += " " + stripped
    return docs


def _class_info(cls: type) -> Dict[str, Any]:
    """Describe a class: where it is defined, its docstring, and where its ``forward`` lives."""
    info: Dict[str, Any] = {
        "qualname": cls.__qualname__,
        "module": cls.__module__,
        "bases": [base.__qualname__ for base in cls.__bases__],
        "mro": [k.__qualname__ for k in cls.__mro__ if k is not object],
        "mro_keys": [f"{k.__module__}.{k.__qualname__}" for k in cls.__mro__ if k is not object],
        "docstring": (inspect.getdoc(cls) or "")[:2500],
        "arg_docs": _parse_arg_docs(cls),
    }
    try:
        source_file = inspect.getsourcefile(cls)
        _, def_line = inspect.getsourcelines(cls)
        info["file"] = _relpath(source_file) if source_file else None
        info["line"] = int(def_line)
    except (OSError, TypeError):
        info["file"], info["line"] = None, None
    forward = getattr(cls, "forward", None)
    owner = None
    for klass in cls.__mro__:
        if "forward" in klass.__dict__:
            owner = klass
            break
    # nn.Module's own ``forward`` is the unimplemented placeholder: a class that is driven through
    # another method (HorizonDecoderCore.decode) has no forward, and must not be shown as having one.
    if owner is nn.Module:
        owner = None
    if forward is not None and owner is not None:
        try:
            lines, start = inspect.getsourcelines(owner.__dict__["forward"])
            info["forward_owner"] = owner.__qualname__
            info["forward_file"] = _relpath(inspect.getsourcefile(owner.__dict__["forward"]) or "")
            info["forward_line"] = int(start)
            info["forward_source"] = "".join(lines[:120])
        except (OSError, TypeError):
            pass
    return info


def describe_modules(model: nn.Module) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[int, str]]:
    """Walk the module tree and describe every module once, by its primary path.

    Args:
        model: The constructed model.

    Returns:
        ``(modules, classes, module_paths)`` -- module records keyed by dotted path, class records
        keyed by qualified class name, and ``id(module) -> primary path``.
    """
    module_paths: Dict[int, str] = {}
    aliases: Dict[str, List[str]] = defaultdict(list)
    for name, module in model.named_modules(remove_duplicate=False):
        if id(module) in module_paths:
            aliases[module_paths[id(module)]].append(name)
        else:
            module_paths[id(module)] = name

    # A parameter is charged to the path where ``named_parameters`` first sees it, so the budget
    # partitions the total: ``decoder.core`` shares ``horizon_core``'s tensors and must not count
    # them twice.
    exclusive: Dict[str, int] = defaultdict(int)
    exclusive_trainable: Dict[str, int] = defaultdict(int)
    for name, parameter in model.named_parameters():
        owner = name.rsplit(".", 1)[0] if "." in name else ""
        path = owner
        while True:
            exclusive[path] += parameter.numel()
            if parameter.requires_grad:
                exclusive_trainable[path] += parameter.numel()
            if path == "":
                break
            path = path.rsplit(".", 1)[0] if "." in path else ""

    classes: Dict[str, Any] = {}
    modules: Dict[str, Any] = {}
    for module_id, path in module_paths.items():
        module = None
        for candidate_name, candidate in model.named_modules(remove_duplicate=False):
            if candidate_name == path:
                module = candidate
                break
        assert module is not None
        cls = type(module)
        key = f"{cls.__module__}.{cls.__qualname__}"
        if key not in classes:
            classes[key] = _class_info(cls)
        own_params = [(n, p) for n, p in module.named_parameters(recurse=False)]
        own_buffers = [(n, b) for n, b in module.named_buffers(recurse=False)]
        ctor = module.__dict__.get("_arch_ctor")
        depth = 0 if path == "" else path.count(".") + 1
        children = [
            module_paths[id(child)]
            for _, child in module.named_children()
            if id(child) in module_paths
        ]
        modules[path] = {
            "path": path,
            "class": cls.__qualname__,
            "class_key": key,
            "family": _module_family(module),
            "depth": depth,
            "parent": None if path == "" else (path.rsplit(".", 1)[0] if "." in path else ""),
            "children": children,
            "aliases": aliases.get(path, []),
            "is_leaf": len(children) == 0,
            "params_total": sum(p.numel() for p in module.parameters()),
            "params_trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "params_exclusive": exclusive.get(path, 0),
            "params_exclusive_trainable": exclusive_trainable.get(path, 0),
            "params_own": [
                {"name": n, "shape": _shape_of(p), "numel": p.numel(),
                 "trainable": bool(p.requires_grad)}
                for n, p in own_params
            ],
            "buffers_own": [
                {"name": n, "shape": _shape_of(b), "dtype": _dtype_of(b)} for n, b in own_buffers
            ],
            "buffers_total": sum(1 for _ in module.buffers()),
            "extra_repr": module.extra_repr(),
            "ctor": ctor,
        }
    return modules, classes, module_paths


# ---------------------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------------------
def _named_outputs(outputs: Any) -> Dict[str, torch.Tensor]:
    """Name whatever ``forward`` returned: a dict keeps its keys, a tensor is ``output``, a tuple
    (or namedtuple) is numbered (or takes its field names); nested containers are flattened.

    Args:
        outputs: The forward's return value.

    Returns:
        ``name -> tensor`` in a stable order.
    """
    if isinstance(outputs, torch.Tensor):
        return {"output": outputs}
    if isinstance(outputs, dict):
        named: Dict[str, torch.Tensor] = {}
        for key, value in outputs.items():
            for sub, tensor in _named_outputs(value).items():
                named[str(key) if sub == "output" else f"{key}.{sub}"] = tensor
        return named
    fields = getattr(outputs, "_fields", None)
    if isinstance(outputs, (tuple, list)):
        named = {}
        for index, value in enumerate(outputs):
            name = fields[index] if fields and index < len(fields) else f"output_{index}"
            for sub, tensor in _named_outputs(value).items():
                named[name if sub == "output" else f"{name}.{sub}"] = tensor
        return named
    return {}

def _try_fx_and_export(model: nn.Module, args: Tuple[torch.Tensor, ...], kwargs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Attempt the two standard tracers and report, honestly, why each fails on this model."""
    report: Dict[str, Any] = {}
    try:
        torch.fx.symbolic_trace(model)
        report["fx_symbolic_trace"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001 - the point is to record the failure
        report["fx_symbolic_trace"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    # torch.export prints its partial graph to stderr on failure; that noise is not a result.
    stderr, sys.stderr = sys.stderr, io.StringIO()
    try:
        torch.export.export(model, args, kwargs=kwargs, strict=False)
        report["torch_export"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        report["torch_export"] = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    finally:
        sys.stderr = stderr
    return report


def _prune_empty_scopes(scopes: Dict[str, Any], root: str) -> None:
    """Drop scopes whose subtree recorded no call (``ModuleList.__iter__`` and the like)."""
    def count(sid: str) -> int:
        record = scopes[sid]
        total = len(record["ops"]) + sum(count(child) for child in record["children"])
        record["ops_total"] = total
        return total

    count(root)
    # An empty scope's whole subtree is empty (the count is over the subtree), so every empty
    # scope can be deleted outright and the surviving child lists filtered afterwards.
    for sid in list(scopes):
        if sid != root and scopes[sid]["ops_total"] == 0:
            del scopes[sid]
    for record in scopes.values():
        record["children"] = [c for c in record["children"] if c in scopes]


def _scope_signature(scopes: Dict[str, Any], ops: Dict[str, Any], tensors: Dict[str, Any], sid: str) -> str:
    """Hash a scope's structure so identical siblings (the ``x6`` block stack) can be found."""
    record = scopes[sid]
    parts: List[str] = [record["kind"], record.get("module_class", ""), record.get("function", "") or ""]
    for oid in record["ops"]:
        op = ops[oid]
        parts.append(op["op"])
        parts.append(",".join(str(tensors[t]["shape"]) for t in op["outputs"]))
    for child in record["children"]:
        parts.append(_scope_signature(scopes, ops, tensors, child))
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]
    record["signature"] = digest
    return digest


def _scope_boundary(scopes: Dict[str, Any], ops: Dict[str, Any], tensors: Dict[str, Any]) -> None:
    """Compute each scope's input/output tensors: what crosses its boundary."""
    membership: Dict[str, set] = {}

    def collect(sid: str) -> set:
        own = set(scopes[sid]["ops"])
        for child in scopes[sid]["children"]:
            own |= collect(child)
        membership[sid] = own
        return own

    roots = [sid for sid, record in scopes.items() if record["parent"] is None]
    for root in roots:
        collect(root)
    for sid, member_ops in membership.items():
        inputs: List[str] = []
        outputs: List[str] = []
        seen_in: set = set()
        seen_out: set = set()
        for oid in sorted(member_ops, key=lambda o: ops[o]["seq"]):
            for entry in ops[oid]["inputs"]:
                if entry["kind"] != "tensor":
                    continue
                tid = entry["id"]
                producer = tensors[tid]["producer"]
                if (producer is None or producer not in member_ops) and tid not in seen_in:
                    seen_in.add(tid)
                    inputs.append(tid)
            for tid in ops[oid]["outputs"]:
                consumers = tensors[tid]["consumers"]
                escapes = any(c not in member_ops for c in consumers) or tensors[tid].get("is_output")
                if escapes and tid not in seen_out:
                    seen_out.add(tid)
                    outputs.append(tid)
        scopes[sid]["inputs"] = inputs
        scopes[sid]["outputs"] = outputs
        scopes[sid]["flops"] = sum((ops[o]["flops"] or 0) for o in member_ops)
        scopes[sid]["activation_bytes"] = sum(
            tensors[t]["bytes"] for o in member_ops for t in ops[o]["outputs"] if not tensors[t]["is_view"]
        )


def main(
    config: Optional[str] = None,
    shard: Optional[str] = None,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    output: Optional[str] = None,
    mode: Optional[str] = None,
) -> int:
    """Build the production model, trace one forward pass and write ``arch.json``.

    Args:
        config: Training YAML; defaults to the package's ``configs/default.yaml``.
        shard: Causal HDF5 shard for the warm-up resolver; defaults to the committed fixture.
        batch_size: Batch size $B$ of the traced forward. Defaults to $8$.
        seed: Global seed for construction and the traced draw. Defaults to $0$.
        output: Where to write the JSON. Defaults to ``arch.json`` beside this file.
        mode: ``eval`` (default) or ``train`` -- which module mode the forward is traced in.

    Returns:
        ``0`` on success, ``1`` if a sanity check failed.
    """
    started = time.time()
    # Repo-root-anchored defaults, so ``main()`` behaves the same from a test or from the CLI.
    config = config or os.path.join(_REPO_ROOT, DEFAULT_CONFIG)
    shard = shard or os.path.join(_REPO_ROOT, DEFAULT_SHARD)
    batch_size = int(batch_size or 8)
    seed = int(seed if seed is not None else 0)
    output = output or os.path.join(os.path.dirname(_THIS_FILE), "arch.json")
    mode = mode or "eval"
    if mode not in ("eval", "train"):
        print(f"mode must be 'eval' or 'train', got {mode!r}")
        return 1

    # 1. Constructor capture, then the model.
    wrapped = install_ctor_capture()
    kwargs, resolved_config, budget_summary = build_model_kwargs(config, shard)
    torch.manual_seed(seed)
    model = MODEL_CLASS(**kwargs)
    model.train(mode == "train")
    print(f"[build] {wrapped} nn.Module classes instrumented; model built from {config}")
    print(f"[build] {budget_summary}")

    # 2. Registries.
    modules, classes, module_paths = describe_modules(model)
    param_names = {id(p): n for n, p in model.named_parameters()}
    buffer_names = {id(b): n for n, b in model.named_buffers()}
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(1 for _ in model.buffers())
    buffer_elements = sum(b.numel() for b in model.buffers())

    # 3. Inputs, from the model-specific hook; the call is ``model(*fwd_args, **fwd_kwargs)``.
    inputs, positional, keyword, meanings = build_inputs(model, batch_size, seed + 1)
    fwd_args = tuple(inputs[name] for name in positional)
    fwd_kwargs = {name: inputs[name] for name in keyword}
    input_spec = {
        name: {"shape": _shape_of(t), "dtype": _dtype_of(t), "meaning": meanings.get(name, "")}
        for name, t in inputs.items()
    }

    # 4. The two standard tracers, for the record.
    tracer_report = _try_fx_and_export(model, fwd_args, fwd_kwargs)

    # 5. Forward hooks for a cross-check of the frame-derived call counts and per-call I/O.
    hook_calls: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    handles = []

    def make_hook(path: str):
        def hook(module, args, kwargs, out):
            flat_in = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
            flat_out = [t for t in tree_flatten(out)[0] if isinstance(t, torch.Tensor)]
            hook_calls[path].append({
                "inputs": [{"shape": _shape_of(t), "dtype": _dtype_of(t)} for t in flat_in],
                "outputs": [{"shape": _shape_of(t), "dtype": _dtype_of(t)} for t in flat_out],
            })
        return hook

    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(make_hook(name), with_kwargs=True))

    # 6. The trace.
    tracer = ArchTracer(model, module_paths, param_names, buffer_names)
    input_ids = {name: tracer.register_input(name, t) for name, t in inputs.items()}
    torch.manual_seed(seed + 2)
    trace_started = time.time()
    with torch.no_grad(), tracer:
        outputs = _named_outputs(model(*fwd_args, **fwd_kwargs))
    trace_seconds = time.time() - trace_started
    for handle in handles:
        handle.remove()

    output_ids: Dict[str, Optional[str]] = {}
    for name, tensor in outputs.items():
        tid = tracer.lookup(tensor)
        output_ids[name] = tid
        if tid is not None:
            tracer.tensors[tid]["is_output"] = True
            tracer.tensors[tid].setdefault("output_names", []).append(name)

    # 7. Measured FLOPs per module, same inputs and seed. The CPU attention kernel is registered
    #    first, or the count would silently omit every attention block.
    _register_cpu_sdpa_flops()
    torch.manual_seed(seed + 2)
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(*fwd_args, **fwd_kwargs)
    measured: Dict[str, int] = {}
    prefix = type(model).__name__
    for key, per_op in counter.get_flop_counts().items():
        if key == "Global":
            continue
        path = "" if key == prefix else key[len(prefix) + 1:] if key.startswith(prefix + ".") else key
        measured[path] = int(sum(per_op.values()))
    for path, record in modules.items():
        record["flops_measured"] = measured.get(path)
        record["flops_measured_from_children"] = False
    # Deepest first, so a rolled-up child is available to its parent.
    for path in sorted(modules, key=lambda p: -modules[p]["depth"]):
        record = modules[path]
        if record["flops_measured"] is None and record["children"]:
            child_values = [modules[c]["flops_measured"] for c in record["children"] if modules[c]["flops_measured"] is not None]
            if child_values:
                record["flops_measured"] = int(sum(child_values))
                record["flops_measured_from_children"] = True
    flops_measured_total = int(counter.get_total_flops())

    # 8. Post-processing of the trace.
    scopes = tracer.scopes
    ops = {op["id"]: op for op in tracer.ops}
    tensors = tracer.tensors
    _prune_empty_scopes(scopes, tracer.root_scope)
    module_by_path = {path: module for path, module in [(module_paths[id(m)], m) for _, m in model.named_modules(remove_duplicate=False) if id(m) in module_paths]}
    for record in scopes.values():
        if "module" in record:
            record["module_class"] = modules[record["module"]]["class"]
            record["params_total"] = modules[record["module"]]["params_total"]
            if record["kind"] == "method":
                # A directly invoked method (decode, build_lag_mask): cite and show *its* source.
                method = getattr(type(module_by_path[record["module"]]), record["method"], None)
                try:
                    lines, start = inspect.getsourcelines(method)
                    record["method_source"] = {
                        "file": _relpath(inspect.getsourcefile(method) or ""),
                        "line": int(start),
                        "source": "".join(lines[:120]),
                    }
                except (OSError, TypeError):
                    record["method_source"] = None
    _scope_signature(scopes, ops, tensors, tracer.root_scope)
    _scope_boundary(scopes, ops, tensors)
    for path in modules:
        modules[path]["invocations"] = [
            sid for sid, record in scopes.items() if record.get("module") == path
        ]
        modules[path]["hook_calls"] = hook_calls.get(path, [])

    # Edges: one per (producer op, consumer op, tensor); model inputs are producers too.
    edges: List[Dict[str, Any]] = []
    for tid, meta in tensors.items():
        source = meta["producer"] if meta["producer"] is not None else (
            f"input:{meta['name']}" if meta["role"] == "input" else f"untraced:{tid}"
        )
        for consumer in meta["consumers"]:
            edges.append({"from": source, "to": consumer, "tensor": tid})

    # Depth of the dataflow DAG: longest chain of ops, computed in execution order.
    longest: Dict[str, int] = {}
    for op in tracer.ops:
        best = 0
        for entry in op["inputs"]:
            if entry["kind"] == "tensor":
                producer = tensors[entry["id"]]["producer"]
                if producer is not None:
                    best = max(best, longest[producer])
        longest[op["id"]] = best + 1
    dag_depth = max(longest.values()) if longest else 0
    module_tree_depth = max(record["depth"] for record in modules.values())

    activation_bytes = sum(t["bytes"] for t in tensors.values() if t["role"] == "activation" and not t["is_view"])
    activation_bytes_incl_views = sum(t["bytes"] for t in tensors.values() if t["role"] == "activation")
    flops_estimated_total = sum((op["flops"] or 0) for op in tracer.ops)
    flops_estimated_matmul = sum((op["flops"] or 0) for op in tracer.ops if op["flops_kind"] == "matmul")
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # 9. Sanity checks, printed and stored.
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        print(f"[check] {'OK ' if ok else 'FAIL'} {name}: {detail}")

    root_record = modules[""]
    check("params: module tree == model.parameters()",
          root_record["params_total"] == total_params,
          f"{root_record['params_total']:,} vs {total_params:,}")
    check("params: exclusive partition sums to the total",
          root_record["params_exclusive"] == total_params,
          f"{root_record['params_exclusive']:,} vs {total_params:,}")
    top_level_sum = sum(modules[c]["params_exclusive"] for c in root_record["children"])
    check("params: top-level exclusive segments sum to the total",
          top_level_sum + sum(p["numel"] for p in root_record["params_own"]) == total_params,
          f"{top_level_sum:,} (+ own {sum(p['numel'] for p in root_record['params_own']):,}) vs {total_params:,}")
    check("trainable: module tree == model.parameters()",
          root_record["params_trainable"] == total_trainable,
          f"{root_record['params_trainable']:,} vs {total_trainable:,}")
    mismatched = []
    for module_id, path in module_paths.items():
        frame_calls = tracer._module_forward_calls.get(module_id, 0)
        if frame_calls != len(modules[path]["hook_calls"]):
            mismatched.append((path, frame_calls, len(modules[path]["hook_calls"])))
    check("scopes: frame-derived forward invocations == hook call counts",
          not mismatched, f"{len(mismatched)} mismatches {mismatched[:5]}")
    check("trace: no untraced tensor inputs", tracer.untraced_inputs == 0,
          f"{tracer.untraced_inputs} untraced")
    check("trace: every op has a repository call site",
          all(op["site"] is not None for op in tracer.ops),
          f"{sum(1 for op in tracer.ops if op['site'] is None)} without a site")
    check("trace: every forward output was produced by a recorded call",
          all(tid is not None for tid in output_ids.values()),
          f"{[n for n, t in output_ids.items() if t is None]} unresolved")
    ratio = flops_estimated_matmul / flops_measured_total if flops_measured_total else float("nan")
    check("flops: estimated matmul-class within 2% of torch.utils.flop_counter",
          abs(ratio - 1.0) < 0.02,
          f"estimated {flops_estimated_matmul:,} vs measured {flops_measured_total:,} (ratio {ratio:.4f})")

    ok = all(item["ok"] for item in checks)

    # 10. Assemble and write.
    kwargs_json = {k: _jsonable(v) for k, v in kwargs.items()}
    model_class_key = f"{type(model).__module__}.{type(model).__qualname__}"
    package_doc = (inspect.getmodule(type(model)).__doc__ or "")[:4000]
    for klass in type(model).__mro__:
        if klass is object or klass is nn.Module:
            continue
        key = f"{klass.__module__}.{klass.__qualname__}"
        if key not in classes:
            classes[key] = _class_info(klass)

    arch = {
        "meta": {
            "generated_by": _relpath(_THIS_FILE),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "device": "cpu",
            "mode": mode,
            "seed": seed,
            "config_path": config,
            "shard_path": shard,
            "budget_summary": budget_summary,
            "trace_seconds": round(trace_seconds, 3),
            "trace_method": (
                "TorchFunctionMode (one node per source-level torch call, real tensors) + "
                "sys.setprofile frame scopes (module/method/function attribution) + tensor-identity "
                "edges (WeakTensorKeyDictionary); forward hooks and torch.utils.flop_counter as "
                "cross-checks"
            ),
            "trace_limitations": [
                "Traced in eval mode: dropout is an identity pass-through node and appears as such.",
                "The trace covers forward() only; compute_loss and the objective are not part of the graph.",
                "Per-op FLOPs are estimates from shapes (flops_kind says which formula); per-module "
                "flops_measured is torch.utils.flop_counter and counts matmul/conv/attention kernels only.",
                "activation_bytes sums every materialised (non-view) intermediate; it is an upper bound "
                "on peak activation memory, not a liveness-aware peak.",
                "Constructor 'site' is the innermost repository frame at construction; the model itself "
                "was constructed by this tool and has none.",
            ],
            "standard_tracers": tracer_report,
            "skipped_non_tensor_calls": tracer.skipped_non_tensor_calls,
        },
        "model": {
            "class": type(model).__qualname__,
            "class_key": model_class_key,
            "package_doc": package_doc,
            "kwargs": kwargs_json,
            "config_excerpt": config_excerpt(resolved_config),
            "geometry": {k: _jsonable(v) for k, v in model_geometry(model).items()},
        },
        "inputs": {name: dict(spec, tensor=input_ids[name]) for name, spec in input_spec.items()},
        "outputs": {name: {"tensor": tid, "shape": _shape_of(t), "dtype": _dtype_of(t)}
                    for (name, t), tid in zip(outputs.items(), output_ids.values())},
        "totals": {
            "params": total_params,
            "params_trainable": total_trainable,
            "params_frozen": total_params - total_trainable,
            "param_bytes": param_bytes,
            "buffers": total_buffers,
            "buffer_elements": buffer_elements,
            "modules": len(modules),
            "leaf_modules": sum(1 for m in modules.values() if m["is_leaf"]),
            "ops": len(tracer.ops),
            "tensors": len(tensors),
            "edges": len(edges),
            "scopes": len(scopes),
            "batch_size": batch_size,
            "activation_bytes": activation_bytes,
            "activation_bytes_incl_views": activation_bytes_incl_views,
            "flops_estimated": flops_estimated_total,
            "flops_estimated_matmul": flops_estimated_matmul,
            "flops_measured": flops_measured_total,
            "dag_depth": dag_depth,
            "module_tree_depth": module_tree_depth,
        },
        "checks": checks,
        "modules": modules,
        "classes": classes,
        "scopes": scopes,
        "root_scope": tracer.root_scope,
        "ops": tracer.ops,
        "tensors": tensors,
        "edges": edges,
        "sites": _SITES,
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(arch, handle, separators=(",", ":"))
    size_mb = os.path.getsize(output) / 1e6
    html_path = os.path.join(os.path.dirname(os.path.abspath(output)), "arch_viz.html")
    inlined = _inline_json_into_html(html_path, arch)

    # 11. Summary.
    print()
    print(f"model            {type(model).__qualname__}  ({mode} mode, seed {seed})")
    print(f"params           {total_params:,} total, {total_trainable:,} trainable, "
          f"{total_params - total_trainable:,} frozen; {param_bytes / 1e6:.2f} MB")
    print(f"buffers          {total_buffers} ({buffer_elements:,} elements)")
    print(f"modules          {len(modules)} ({sum(1 for m in modules.values() if m['is_leaf'])} leaves), tree depth {module_tree_depth}")
    print(f"trace            {len(tracer.ops)} ops, {len(tensors)} tensors, {len(edges)} edges, "
          f"{len(scopes)} scopes, DAG depth {dag_depth}, {trace_seconds:.2f}s")
    print(f"flops @B={batch_size}  measured {flops_measured_total / 1e9:.3f} GFLOP (matmul/conv/sdpa), "
          f"estimated {flops_estimated_total / 1e9:.3f} GFLOP all ops")
    print(f"activations      {activation_bytes / 1e6:.1f} MB materialised (+{(activation_bytes_incl_views - activation_bytes) / 1e6:.1f} MB views)")
    print(f"inputs           " + ", ".join(f"{k}{v['shape']}" for k, v in input_spec.items()))
    first_output = next(iter(outputs.items()))
    print(f"outputs          {len(outputs)} tensors, e.g. {first_output[0]}{_shape_of(first_output[1])}")
    print(f"budget bar       " + ", ".join(
        f"{c}={modules[c]['params_exclusive']:,}" for c in root_record["children"] if modules[c]["params_exclusive"]))
    print(f"wrote            {_relpath(output)} ({size_mb:.2f} MB) in {time.time() - started:.1f}s")
    print(f"tracers          fx={tracer_report['fx_symbolic_trace']['ok']}, export={tracer_report['torch_export']['ok']}")
    print("html             " + (f"refreshed {_relpath(html_path)}" if inlined
                                 else "no arch_viz.html beside the JSON; nothing inlined"))
    return 0 if ok else 1


#: Markers of the JSON block inside ``arch_viz.html``. The page reads the JSON from this script
#: element and nowhere else, so refreshing the page is replacing what sits between them.
_HTML_JSON_OPEN = '<script id="arch-json" type="application/json">'
_HTML_JSON_CLOSE = "</script>"


def _inline_json_into_html(html_path: str, arch: Dict[str, Any]) -> bool:
    """Replace the inlined JSON inside ``arch_viz.html`` with ``arch``, if the page exists.

    ``</`` inside a string is escaped as ``<\\/`` (valid JSON, inert in HTML), so a source
    snippet can never close the script element early.

    Args:
        html_path: Path of the page.
        arch: The architecture record just written to ``arch.json``.

    Returns:
        ``True`` if the page was rewritten, ``False`` if it does not exist or has no JSON block.
    """
    if not os.path.isfile(html_path):
        return False
    with open(html_path, "r", encoding="utf-8") as handle:
        html = handle.read()
    open_at = html.find(_HTML_JSON_OPEN)
    if open_at < 0:
        return False
    body_at = open_at + len(_HTML_JSON_OPEN)
    close_at = html.find(_HTML_JSON_CLOSE, body_at)
    if close_at < 0:
        return False
    payload = json.dumps(arch, separators=(",", ":")).replace("</", "<\\/")
    rewritten = html[:body_at] + "\n" + payload + "\n" + html[close_at:]
    with open(html_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(rewritten)
    return True


# ---------------------------------------------------------------------------------------
# Command line / Run button
# ---------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the parser. No ``required`` and no non-``None`` defaults: see the launch convention."""
    parser = argparse.ArgumentParser(description="Extract the traced architecture into arch.json.")
    parser.add_argument("--config", dest="config", default=None,
                        help="Training YAML (default: this package's configs/default.yaml)")
    parser.add_argument("--shard", dest="shard", default=None,
                        help="Causal HDF5 shard for the warm-up resolver (default: the committed fixture)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                        help="Batch size of the traced forward (default: 8)")
    parser.add_argument("--seed", dest="seed", type=int, default=None, help="Seed (default: 0)")
    parser.add_argument("--output", dest="output", default=None,
                        help="Output JSON path (default: arch.json beside this file)")
    parser.add_argument("--mode", dest="mode", choices=("eval", "train"), default=None,
                        help="Module mode the forward is traced in (default: eval)")
    return parser


#: Launch values for an IDE's Run button, keyed by argparse ``dest``. Every key is optional; a
#: ``None`` means "use the documented default". A command-line flag overrides the matching key.
RUN_ARGS: Dict[str, Any] = {
    "config": None,       # e.g. "teb_vae/lag_attn_transformer_cfs/configs/default.yaml"
    "shard": None,        # e.g. "teb_vae/lag_attn/tests/fixtures/tiny_shard_causal.hdf5"
    "batch_size": None,   # e.g. 8
    "seed": None,         # e.g. 0
    "output": None,       # e.g. "teb_vae/lag_attn_transformer_cfs/nets/arch_viz/arch.json"
    "mode": None,         # "eval" or "train"
}


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    """Merge the command line over ``RUN_ARGS`` and run from the repository root."""
    parser = build_parser()
    values, sources = resolve_launch_args(parser, RUN_ARGS, argv)
    os.chdir(_REPO_ROOT)
    print("[launch] " + ", ".join(f"{k}={values[k]!r} ({sources[k]})" for k in sorted(values)))
    return main(**values)


if __name__ == "__main__":
    sys.exit(_cli())
