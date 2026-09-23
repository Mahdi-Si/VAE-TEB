r"""Where a training step may not synchronise the host with the device.

A synchronising call (``.item()``, ``bool(tensor)``, a pageable host-to-device copy) makes the CPU
wait until the GPU has drained everything queued so far, which throws away the run-ahead that
hides launch overhead. The sites below were made sync-free without moving a number, and this is
what keeps them that way: one train-stage step runs under CUDA's synchronisation debug mode, every
warning it raises is attributed to the innermost repository frame that issued it, and none of the
named functions may be that frame. No total count is pinned, because the deferred sites and future
torch versions would move it.

The attribution test needs a CUDA device and skips without one, stating so. The transfer test
runs on the ``meta`` device and needs nothing.
"""
from __future__ import annotations

import re
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch

from hdf5_dataset.hdf5_dataset import AttributeDict
from teb_vae.lag_attn_rws.nets import losses, raw_masks

from .conftest import make_stub_batch, make_task

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Functions in whose own frame no synchronising operation may originate during a train step.
#: ``_validate_anchors`` is not among them: its two refusals are device reads by nature, and the
#: objective keeps exactly one validation per step, which the call count below pins instead.
SYNC_FREE_FUNCTIONS = frozenset(
    {
        "anchor_phase",
        "_as_float",
        "_phase_field",
        "_build_anchor_index",
        "_refuse_phase_outside",
        "masked_raw_likelihood",
    }
)

#: The objective's echoed weights: the lines of ``compute_loss`` that build them may not copy from
#: the host. The rest of ``compute_loss`` keeps its deferred diagnostics, so only these lines are
#: attributed rather than the whole function.
ECHO_KEYS = ("kld_beta", "beta_prior", "lambda_ms", "lambda_deriv", "lambda_boundary")


def _echo_lines() -> Tuple[str, List[int]]:
    """The file and line numbers of the five echo entries in ``compute_loss``."""
    path = Path(losses.__file__)
    pattern = re.compile(r'"(' + "|".join(ECHO_KEYS) + r')":')
    lines = [
        number
        for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(text)
    ]
    assert len(lines) == len(ECHO_KEYS), "expected one echo line per key"
    return str(path.resolve()), lines


def _batch_on(device: torch.device) -> AttributeDict:
    """The stub batch as the loader's mapping type, moved by the task's own transfer."""
    task = make_task()
    stub = make_stub_batch()
    batch = AttributeDict(vars(stub))
    return task.transfer_batch_to_device(batch, device, dataloader_idx=0)


def test_the_transfer_moves_every_tensor_but_epoch() -> None:
    """The phase is hashed on the host, so ``epoch`` stays there; nothing else does."""
    moved = _batch_on(torch.device("meta"))
    assert moved["epoch"].device.type == "cpu"
    for key, value in moved.items():
        if key == "epoch" or not isinstance(value, torch.Tensor):
            continue
        assert value.device.type == "meta", key
    assert isinstance(moved["guid"], list)


def test_a_phase_outside_the_stride_is_refused_from_either_device() -> None:
    """The range refusal moved to the host and must still fire there, and on the device."""
    model = make_task().orig_model
    stride = int(model.anchor_stride)
    bad = torch.tensor([0, stride], dtype=torch.long)
    with pytest.raises(ValueError, match=f"anchor_phase {stride} is outside"):
        model._build_anchor_index(2, torch.device("cpu"), bad, stride)
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match=f"anchor_phase {stride} is outside"):
            model._build_anchor_index(2, torch.device("cuda"), bad.cuda(), stride)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync attribution needs a CUDA device")
def test_a_train_step_synchronises_nowhere_in_the_named_sites(monkeypatch) -> None:
    """Attribute every synchronising operation of one train step to the innermost repository
    frame; none of the named functions, and none of the echo lines, may be that frame. The anchor
    set is validated exactly once per step."""
    device = torch.device("cuda")
    task = make_task().to(device).train()
    batch = _batch_on(device)

    calls = Counter()
    original_validate = raw_masks._validate_anchors

    def counting_validate(*args, **kwargs):
        calls["_validate_anchors"] += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(raw_masks, "_validate_anchors", counting_validate)

    records: List[Tuple[str, traceback.StackSummary]] = []

    def capture(message, category, filename, lineno, file=None, line=None):
        records.append((str(message), traceback.extract_stack()))

    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode(1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = capture
            loss, _metrics = task.compute_loss_and_metrics(batch, 0, "train")
            loss.backward()
    finally:
        torch.cuda.set_sync_debug_mode(0)
        torch.cuda.synchronize()

    echo_file, echo_lines = _echo_lines()
    offenders: Dict[str, int] = Counter()
    for message, stack in records:
        if "synchroniz" not in message.lower():
            continue
        # The innermost frame inside the repository: the line that issued the synchronising call.
        # This file's own frames (the test, and the capture hook the warning lands in) are not it.
        repo_frames = [
            frame
            for frame in stack
            if str(Path(frame.filename).resolve()).startswith(str(_REPO_ROOT))
            and ".venv" not in frame.filename
            and Path(frame.filename).resolve() != Path(__file__).resolve()
        ]
        if not repo_frames:
            continue
        innermost = repo_frames[-1]
        if innermost.name in SYNC_FREE_FUNCTIONS:
            offenders[f"{innermost.name}:{innermost.lineno}"] += 1
        if (
            str(Path(innermost.filename).resolve()) == echo_file
            and innermost.lineno in echo_lines
        ):
            offenders[f"compute_loss echo:{innermost.lineno}"] += 1

    assert not offenders, f"synchronising operations attributed to sync-free sites: {dict(offenders)}"
    assert calls["_validate_anchors"] == 1, calls
