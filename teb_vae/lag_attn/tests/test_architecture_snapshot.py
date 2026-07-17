"""The flatten changed the code's shape and not the model's. This is the proof.

This model was rebuilt out of a three-level inheritance chain into one readable class. The
rebuild is not checkpoint-compatible and is not claimed to be bit-identical, so what has to be
established is *structural*: the same modules, the same shapes, the same parameter count, the
same forward and loss contracts.

The fixtures under ``fixtures/`` are snapshots of the original, captured before any of this
existed (regenerable with ``scripts/capture_v3_architecture.py``). ``_DEVIATIONS`` below is the
complete list of intentional differences, and it is the single source of truth for what the
flatten changed. Everything not in that list must match exactly.

**Two geometries are checked, and the second is the one that matters.** The original's own test
suite used a flag set it called "prod" which leaves four flags the shipped config sets at their
constructor defaults -- so a snapshot of it records none of the per-head posterior, the FiLM
generator, the third refine block, or the extra encoder conv blocks, and none of the roughly
145k parameters they carry. Asserting only that geometry would prove the rebuild reproduces a
model nobody trains. ``shipped`` closes that gap.

The map is not a place to record surprises. If an assertion here fails, the answer is to find
the bug in the rebuild -- widening the map to make the test green would delete the only evidence
that the rebuild is the same model, which is the entire reason the snapshot exists.

What this cannot catch: a transposed weight, or a reordered residual. Both preserve every shape
and count. That is an accepted limit and the reason the original is forked rather than deleted --
a golden-tensor comparison against it remains available if a retrain ever diverges.

Run with ``-s`` to print the deviation map and the matched counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from teb_vae.lag_attn.nets.model import SeqVaeLagAttn
from teb_vae.lag_attn.tests.conftest import PROD_KWARGS, SHIPPED_KWARGS

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# (variant, fixture filename, the kwargs the rebuild is built with).
_VARIANTS = [
    ("prod", "v3_architecture.json", PROD_KWARGS),
    ("shipped", "v3_architecture_shipped.json", SHIPPED_KWARGS),
]

# Every intentional difference between the original and the rebuild. Each entry says what
# changed and why it is not a defect.
_DEVIATIONS = {
    "encoder_wrappers_removed": (
        "The two encoders were each wrapped in a 31-line class whose only content was a `body` "
        "attribute holding the real encoder. The wrappers existed to shape state-dict keys, "
        "which stopped mattering once checkpoint compatibility was dropped. Their differing "
        "default kernels were never load-bearing: the model computes and passes kernels "
        "explicitly. Effect: `{target,source}_encoder.body.X` -> `{target,source}_encoder.X`. "
        "Parameters unchanged."
    ),
    "latent_stats_buffers_removed": (
        "`mu_post_running_{mean,var,count}` backed a latent-normalisation mechanism with no "
        "consumer in this tree. It was also the only thing in the model that imported loguru or "
        "torch.distributed, the only place a batch field name was read, and it cost four device "
        "syncs per training step. Effect: 3 buffers gone. They are buffers, not parameters, so "
        "the parameter count is unchanged."
    ),
    "raw_future_pred_removed": (
        "The forward dict carried `raw_future_pred: None` -- a non-tensor in a dict of tensors "
        "-- produced by a decoder that was a stub raising NotImplementedError and was never "
        "constructed. Effect: one forward key gone. No parameters were ever involved."
    ),
    "construction_flags_removed": (
        "`posterior_logvar` and `logvar_bound` selected between a parity branch and the shipped "
        "behaviour. The shipped values are now the only behaviour, so the flags are gone and "
        "the snapshot -- captured under the old signature, which required them -- is compared "
        "against a model built without them. Structurally identical: the snapshot was captured "
        "at exactly the values that now hold unconditionally."
    ),
    "dead_lag_bank_removed": (
        "`LagMemoryBankBuilder` was constructed on every model and never called; strided views "
        "over the projected keys replaced it. It held no parameters and no buffers, so its "
        "removal provably cannot perturb this snapshot. Note it also carried the only "
        "`max_lag >= 0` check, which the rebuild re-established in the model's constructor."
    ),
}

_ENCODER_WRAPPER_PREFIXES = ("target_encoder.body.", "source_encoder.body.")
_LATENT_STATS_PREFIX = "mu_post_running_"
_REMOVED_FORWARD_KEYS = ("raw_future_pred",)


@pytest.fixture(params=_VARIANTS, ids=lambda v: v[0])
def variant(request):
    """Yield ``(name, snapshot, rebuilt_model, kwargs)`` for each captured geometry."""
    name, filename, kwargs = request.param
    path = _FIXTURE_DIR / filename
    assert path.is_file(), (
        f"{path} is missing; regenerate it with "
        f"scripts/capture_v3_architecture.py --variant {name} --out {path}"
    )
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["variant"] == name, f"{path} holds variant {snapshot['variant']!r}"

    torch.manual_seed(0)
    model = SeqVaeLagAttn(**dict(kwargs)).eval()
    return name, snapshot, model, dict(kwargs)


def _expected_state_dict(snapshot: dict) -> dict:
    """Apply the intentional deviations to the snapshot's state dict."""
    expected = {}
    for key, shape in snapshot["state_dict"].items():
        if key.startswith(_LATENT_STATS_PREFIX):
            continue  # latent_stats_buffers_removed
        for prefix in _ENCODER_WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = prefix.replace(".body.", ".") + key[len(prefix) :]
                break  # encoder_wrappers_removed
        expected[key] = shape
    return expected


def test_the_deviation_map_is_printed(capsys):
    """The demo: a readable diff rather than a green dot. Run with -s to see it."""
    with capsys.disabled():
        print("\n\nIntentional deviations from the original architecture:\n")
        for name, reason in _DEVIATIONS.items():
            print(f"  * {name}")
            for line in reason.split(". "):
                if line.strip():
                    print(f"      {line.strip().rstrip('.')}.")
            print()


def test_the_two_geometries_are_actually_different(capsys):
    """`shipped` is only worth checking if it builds something `prod` does not.

    If the two fixtures ever collapsed onto the same geometry -- someone "simplifies" the
    variants, or a flag quietly stops gating its module -- the second assertion set would become
    a duplicate of the first, and the production-only structure would go back to being untested
    with no test turning red.
    """
    prod = json.loads((_FIXTURE_DIR / "v3_architecture.json").read_text(encoding="utf-8"))
    shipped = json.loads(
        (_FIXTURE_DIR / "v3_architecture_shipped.json").read_text(encoding="utf-8")
    )

    assert shipped["total_params"] > prod["total_params"]
    assert len(shipped["state_dict"]) > len(prod["state_dict"])

    production_only = set(shipped["state_dict"]) - set(prod["state_dict"])
    with capsys.disabled():
        print(
            f"\n  shipped adds {shipped['total_params'] - prod['total_params']} params and "
            f"{len(production_only)} state-dict keys that `prod` never builds\n"
        )

    # Each of the four flags gates a real, distinct module. Naming them means a flag that
    # silently stopped taking effect is caught here rather than never.
    for probe, gated_by in (
        ("a_head_norm", "head_structured_latent"),
        ("film_gen", "horizon_film"),
        ("refine.blocks.2", "horizon_depth: 3"),
        ("convs.4", "encoder_extra_dilations"),
    ):
        assert any(probe in key for key in production_only), f"{gated_by} gated nothing"

    # freeze_unused_attn_proj is inert without head structure, so only `shipped` can observe it.
    assert prod["trainable_params"] == prod["total_params"]
    assert shipped["trainable_params"] < shipped["total_params"]


def test_total_parameter_count_matches_exactly(variant, capsys):
    name, snapshot, rebuilt, _ = variant
    total = sum(p.numel() for p in rebuilt.parameters())
    with capsys.disabled():
        print(f"  [{name}] parameters: rebuilt {total} vs original {snapshot['total_params']}")
    assert total == snapshot["total_params"]


def test_trainable_parameter_count_matches_exactly(variant):
    _, snapshot, rebuilt, _ = variant
    trainable = sum(p.numel() for p in rebuilt.parameters() if p.requires_grad)
    assert trainable == snapshot["trainable_params"]


def test_every_state_dict_key_maps_one_to_one(variant, capsys):
    name, snapshot, rebuilt, _ = variant
    expected = _expected_state_dict(snapshot)
    actual = {key: list(tensor.shape) for key, tensor in rebuilt.state_dict().items()}

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    with capsys.disabled():
        print(
            f"  [{name}] state dict: {len(snapshot['state_dict'])} original keys -> "
            f"{len(expected)} after deviations; rebuilt has {len(actual)}"
        )

    assert not missing, f"keys the rebuild lost: {missing}"
    assert not unexpected, f"keys the rebuild invented: {unexpected}"


def test_every_shared_key_has_an_identical_shape(variant):
    _, snapshot, rebuilt, _ = variant
    expected = _expected_state_dict(snapshot)
    actual = {key: list(tensor.shape) for key, tensor in rebuilt.state_dict().items()}

    drifted = {
        key: (expected[key], actual[key])
        for key in sorted(set(expected) & set(actual))
        if expected[key] != actual[key]
    }
    assert not drifted, f"shape drift: {drifted}"


def test_the_encoder_wrapper_removal_is_a_pure_rename(variant):
    """Every wrapper key must land on a real key, or the "rename" is hiding a deletion."""
    _, snapshot, rebuilt, _ = variant
    wrapped = [key for key in snapshot["state_dict"] if key.startswith(_ENCODER_WRAPPER_PREFIXES)]
    assert wrapped, "the snapshot records no wrapped encoder keys"

    actual = set(rebuilt.state_dict())
    for key in wrapped:
        renamed = key.replace("_encoder.body.", "_encoder.")
        assert renamed in actual, f"{key} renamed to {renamed}, which does not exist"


def test_only_the_latent_stats_buffers_were_dropped(variant):
    _, snapshot, rebuilt, _ = variant
    dropped = [key for key in snapshot["state_dict"] if key.startswith(_LATENT_STATS_PREFIX)]
    assert sorted(dropped) == [
        "mu_post_running_count",
        "mu_post_running_mean",
        "mu_post_running_var",
    ]
    assert not list(rebuilt.named_buffers()), "the rebuild registers a buffer it should not"


def test_the_forward_contract_matches_minus_the_recorded_removals(variant, inputs):
    _, snapshot, rebuilt, _ = variant
    torch.manual_seed(0)
    with torch.no_grad():
        out = rebuilt(*inputs)

    expected = {
        key: shape
        for key, shape in snapshot["forward_keys"].items()
        if key not in _REMOVED_FORWARD_KEYS
    }
    actual = {key: list(value.shape) for key, value in out.items()}

    assert sorted(actual) == sorted(expected), (
        f"forward keys differ; rebuilt-only={sorted(set(actual) - set(expected))}, "
        f"original-only={sorted(set(expected) - set(actual))}"
    )
    drifted = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    assert not drifted, f"forward shape drift: {drifted}"


def test_the_removed_forward_keys_are_actually_gone(variant, inputs):
    """Guards the map itself: a removal recorded but not performed would go unnoticed."""
    _, snapshot, rebuilt, _ = variant
    for key in _REMOVED_FORWARD_KEYS:
        assert key in snapshot["forward_keys"], f"{key} is recorded as removed but never existed"
    torch.manual_seed(0)
    with torch.no_grad():
        out = rebuilt(*inputs)
    for key in _REMOVED_FORWARD_KEYS:
        assert key not in out


def test_the_loss_contract_matches(variant, inputs):
    _, snapshot, rebuilt, _ = variant
    torch.manual_seed(0)
    with torch.no_grad():
        out = rebuilt(*inputs)
        loss = rebuilt.compute_loss(out, inputs[0], inputs[1])
    assert sorted(loss) == sorted(snapshot["loss_keys"])


def test_the_causalised_norm_count_matches(variant):
    _, snapshot, rebuilt, _ = variant
    assert rebuilt.n_causalized_norms == snapshot["n_causalized_norms"]


def test_the_snapshot_was_captured_at_the_geometry_this_suite_uses(variant, inputs):
    """A snapshot of a differently-shaped model would compare nothing."""
    _, snapshot, _, _ = variant
    geometry = snapshot["input_geometry"]
    y_st, y_ph, u_stream = inputs
    assert y_st.shape == (geometry["batch"], geometry["seq_len"], geometry["c_y_st"])
    assert y_ph.shape == (geometry["batch"], geometry["seq_len"], geometry["c_y_ph"])
    assert u_stream.shape == (geometry["batch"], geometry["seq_len"], geometry["c_u"])


def test_the_snapshot_kwargs_differ_from_the_rebuilds_only_by_the_recorded_flags(variant):
    """Pins the fourth deviation: the two signatures differ by exactly the retired flags."""
    _, snapshot, _, kwargs = variant
    snapshot_kwargs = snapshot["kwargs"]

    only_in_snapshot = set(snapshot_kwargs) - set(kwargs)
    assert only_in_snapshot == {"posterior_logvar", "logvar_bound"}
    # And they were captured at exactly the values that now hold unconditionally.
    assert snapshot_kwargs["posterior_logvar"] == "residual"
    assert snapshot_kwargs["logvar_bound"] == "smooth"
    # Everything else agrees, so nothing else can explain a structural difference. The JSON
    # round-trip turns tuples into lists, so compare on value rather than type.
    for key, value in kwargs.items():
        recorded = snapshot_kwargs[key]
        if isinstance(value, tuple):
            recorded = tuple(recorded)
        assert recorded == value, f"{key} differs: {recorded} vs {value}"
