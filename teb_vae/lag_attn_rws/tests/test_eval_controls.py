r"""The permutation control must pair a target with a *stranger's* source, not a neighbour's.

$\pi(i) \neq i$ is the wrong guarantee on this loader. ``test_dataloader`` is unshuffled over
eight concatenated per-subgroup shards and one delivery contributes tens of consecutive
20-minute segments, so a batch routinely holds several segments of the same recording -- and
Sattolo's algorithm will happily pair one with another. Those two share a mother, a sensor
placement and a labour; the "shuffled" forecast built from one is not the out-of-recording
control the ordering $D_{\mathrm{full}} < D_{\mathrm{base}} < D_{\mathrm{shuffled}}$ is read
against, and the contrast is weakened by an amount nothing reports.

So the draw takes the recording identifiers, and three things are checked here: that the pairing
really does cross recordings, that a batch too concentrated to admit any such pairing is
excluded *and counted* rather than quietly downgraded, and that the ungrouped draw is unchanged
for every caller that does not pass groups.
"""
from __future__ import annotations

from collections import Counter

import pytest
import torch

from teb_vae.lag_attn_rws.eval.metrics import batch_recordings, evaluate, evaluate_batch
from teb_vae.lag_attn_rws.nets import controls

from .conftest import BATCH, make_stub_batch


@pytest.fixture
def trained_task(task, perturb_posterior):
    """A tiny task whose posterior has been moved off the prior, as elsewhere in the suite."""
    module = task()
    perturb_posterior(module.orig_model)
    module.eval()
    return module


class _OneBatchLoader:
    """A dataloader-shaped iterable over a fixed list of batches."""

    def __init__(self, batches):
        self._batches = list(batches)

    def __iter__(self):
        return iter(self._batches)


def _labelled(batch_size: int, guids, seed: int = 0):
    """A stub batch carrying recording identifiers."""
    batch = make_stub_batch(batch_size=batch_size, seed=seed)
    batch.guid = list(guids)
    return batch


# =============================================================================
# The draw itself
# =============================================================================
@pytest.mark.parametrize(
    "groups",
    [
        ["a", "a", "b", "b"],
        ["a", "b"],
        ["a", "a", "b", "c"],
        ["a", "b", "c", "d", "e", "f"],
        ["a"] * 4 + ["b"] * 3 + ["c"] * 1,
    ],
)
def test_a_grouped_derangement_never_pairs_within_a_group(groups):
    """The whole point, over compositions ranging from balanced to right at the feasibility
    boundary. Drawn repeatedly because a construction that is *usually* cross-group would pass a
    single draw."""
    generator = torch.Generator().manual_seed(0)

    for _ in range(50):
        perm = controls.make_derangement(len(groups), generator=generator, groups=groups)

        assert torch.equal(perm.sort().values, torch.arange(len(groups))), "not a permutation"
        for position, partner in enumerate(perm.tolist()):
            assert groups[position] != groups[partner], f"{groups} paired within its own group"


def test_a_grouped_derangement_is_reproducible_and_not_constant():
    """Reproducible so a run's controls are, and varying so it is a draw rather than a fixed
    rotation of whatever order the batch happened to arrive in."""
    groups = ["a", "a", "b", "b", "c", "c"]
    first = controls.make_derangement(6, generator=torch.Generator().manual_seed(3), groups=groups)
    again = controls.make_derangement(6, generator=torch.Generator().manual_seed(3), groups=groups)

    generator = torch.Generator().manual_seed(0)
    seen = {
        tuple(controls.make_derangement(6, generator=generator, groups=groups).tolist())
        for _ in range(50)
    }

    assert torch.equal(first, again)
    assert len(seen) > 1


def test_the_ungrouped_draw_is_untouched():
    """Every caller that passes no groups -- the model's own training-time control among them --
    must get the same permutation it got before, bit for bit."""
    first = controls.make_derangement(16, generator=torch.Generator().manual_seed(7))
    second = controls.make_derangement(16, generator=torch.Generator().manual_seed(7))

    assert torch.equal(first, second)
    assert not bool((first == torch.arange(16)).any())


@pytest.mark.parametrize(
    "groups,expected",
    [
        (["a", "b"], True),
        (["a", "a", "b", "b"], True),
        (["a", "a", "a", "b"], False),
        (["a", "a"], False),
        (["a"], False),
        (["a", "a", "a", "b", "b", "c"], True),
    ],
)
def test_feasibility_is_the_exact_half_batch_condition(groups, expected):
    r"""Hall's theorem gives $2\max_g |g| \le B$ exactly, so the predicate rejects only batches
    that genuinely have no valid pairing -- ``['a','a','a','b']`` has none, and
    ``['a','a','a','b','b','c']`` does."""
    assert controls.groups_can_derange(groups) is expected


def test_an_impossible_grouping_raises_with_the_offending_group_named():
    """A distinct exception type, so a caller can catch exactly this and not a shape error, and
    a message that says which recording filled the batch."""
    with pytest.raises(controls.NoCrossGroupPartner, match="GUID_A"):
        controls.make_derangement(4, groups=["GUID_A", "GUID_A", "GUID_A", "GUID_B"])


def test_a_group_list_of_the_wrong_length_is_refused():
    with pytest.raises(ValueError, match="one label per batch element"):
        controls.make_derangement(4, groups=["a", "b"])


def test_the_forward_control_pairs_across_recordings(trained_task, inputs):
    """Threaded all the way through the rebuild, not merely available on the draw."""
    model = trained_task.orig_model
    y_st, y_ph, u_stream = inputs
    with torch.no_grad():
        outputs = model(y_st, y_ph, u_stream)

    permuted = controls.perm_forward_outputs(model, outputs, groups=["a", "b"])

    assert permuted["perm_index"].tolist() == [1, 0]


# =============================================================================
# The evaluation loop's accounting
# =============================================================================
def test_a_batch_spanning_two_recordings_pairs_across_them(trained_task):
    readout = evaluate_batch(
        trained_task, _labelled(BATCH, ["a", "b"]), num_samples=1
    )

    assert readout.n_control_pairs == BATCH
    assert readout.n_same_recording_pairs == 0


def test_a_single_recording_batch_is_excluded_and_counted(trained_task):
    """Excluded whole rather than scored without its control: a partially scored batch produces
    a different column set, and averaging an inconsistent set together is how a control stops
    being reported with nothing failing. Counted because the batches that one recording fills on
    its own are the *longest* recordings' -- dropping them silently removes a non-random slice.
    """
    loader = _OneBatchLoader(
        [_labelled(2, ["solo", "solo"]), _labelled(2, ["a", "b"], seed=1)]
    )

    results = evaluate(trained_task, loader, num_samples=1)

    assert results["n_batches"] == 1
    assert results["controls"]["n_batches_excluded_no_cross_recording_partner"] == 1
    assert results["controls"]["n_samples_excluded_no_cross_recording_partner"] == 2
    assert results["controls"]["same_recording_pairing_rate"] == 0.0


def test_the_control_statistics_are_reported_even_when_nothing_was_excluded(trained_task):
    """Present at zero, not absent: a reader cannot tell an unreported exclusion count from a
    zero one, and the pairing rate is the only evidence the control is still a control."""
    loader = _OneBatchLoader([_labelled(4, ["a", "a", "b", "b"])])

    control_block = evaluate(trained_task, loader, num_samples=1)["controls"]

    assert control_block["n_batches_excluded_no_cross_recording_partner"] == 0
    assert control_block["n_samples_excluded_no_cross_recording_partner"] == 0
    assert control_block["n_control_pairs"] == 4
    assert control_block["n_same_recording_pairs"] == 0
    assert control_block["same_recording_pairing_rate"] == 0.0


def test_a_batch_with_no_identifiers_still_runs_its_control_and_says_so(trained_task):
    """An absent ``guid`` means the grouping is *unknown*, which is not the same as every sample
    belonging to one recording. The ungrouped derangement still runs -- otherwise every stub
    batch in the suite would be excluded -- and the pairing rate is ``None`` rather than a
    fabricated zero, because nothing was checked."""
    results = evaluate(trained_task, _OneBatchLoader([make_stub_batch()]), num_samples=1)

    assert results["n_batches"] == 1
    assert results["controls"]["n_control_pairs"] == 0
    assert results["controls"]["same_recording_pairing_rate"] is None
    assert results["controls"]["n_batches_excluded_no_cross_recording_partner"] == 0


def test_the_recording_accessor_distinguishes_absent_from_uniform(stub_batch):
    assert batch_recordings(stub_batch, BATCH) is None
    assert batch_recordings(_labelled(2, ["a", "a"]), 2) == ["a", "a"]


def test_a_real_run_reports_its_pairing_rate_and_its_exclusions(evaluated):
    """End to end, through the shards and the serialiser: the accounting is in the artifact, not
    only in the return value."""
    control_block = evaluated["summary"]["results"]["controls"]

    assert control_block["same_recording_pairing_rate"] == 0.0
    assert control_block["n_control_pairs"] == evaluated["summary"]["results"]["n_samples"]
    assert control_block["n_samples_excluded_no_cross_recording_partner"] == 0


# =============================================================================
# Against the real loader, where the pairing problem actually arises
# =============================================================================
def test_the_shuffled_control_pass_breaks_up_a_recordings_consecutive_segments(
    multi_class_loader,
):
    """The reason ``run.py`` re-batches under a fixed-seed shuffle. Written against the real
    generated shards, because the failure is a property of how they are laid out on disk: the
    loader concatenates per-subgroup files and each recording's segments are adjacent in them,
    so a batch no wider than a recording's segment count is *entirely* that recording -- and a
    batch of one recording has no stranger in it to borrow a source from at all.

    Read at a batch size equal to the fixture's segments per recording, which is the smallest
    setting that reproduces the failure; the production ratio is the same one, tens of segments
    against a batch of 32.
    """
    from torch.utils.data import DataLoader

    from teb_vae.lag_attn_rws.eval.run import shuffled_control_loader
    from .conftest import MULTI_CLASS_SEGMENTS_PER_GUID

    def _largest_recording_share(loader):
        return max(max(Counter(batch.guid).values()) / len(batch.guid) for batch in loader)

    unshuffled = DataLoader(
        multi_class_loader.dataset,
        batch_size=MULTI_CLASS_SEGMENTS_PER_GUID,
        shuffle=False,
        collate_fn=multi_class_loader.collate_fn,
    )
    shuffled = shuffled_control_loader(unshuffled, seed=0)

    assert _largest_recording_share(unshuffled) == 1.0, (
        "the unshuffled loader is expected to hand out whole-recording batches at this width; "
        "if it no longer does, this guard is measuring nothing"
    )
    assert _largest_recording_share(shuffled) < 1.0
    assert sum(len(batch.guid) for batch in shuffled) == sum(
        len(batch.guid) for batch in unshuffled
    ), "a shuffle is a reordering; every segment must still be scored exactly once"


def test_the_sample_cap_bounds_the_pass_and_reaches_every_shard(multi_class_loader):
    """``eval_config.max_samples`` is a stratified draw over the whole index space, never a prefix.

    The loader concatenates per-subgroup files, so a prefix cap yields one subgroup and one
    clinical class -- the predecessor's documented "only 1 class found" failure. Stratifying by
    shard guarantees every file appears whenever the cap is at least the shard count, rather than
    making it merely likely.
    """
    from teb_vae.lag_attn_rws.eval.run import capped_sample_loader, dataset_shard_keys

    keys = dataset_shard_keys(multi_class_loader)
    n_total = len(multi_class_loader.dataset)
    cap = len(set(keys)) * 2

    capped, record = capped_sample_loader(multi_class_loader, cap, seed=0)

    assert cap < n_total, "the fixture must be larger than the cap or this tests nothing"
    assert sum(len(batch.guid) for batch in capped) == cap, "the cap has to actually bind"
    assert record == {
        "max_samples": cap,
        "applied": True,
        "n_total": n_total,
        "n_drawn": cap,
        "stratified_by": "source_file_basename",
        "n_shards_drawn": len(set(keys)),
    }
    drawn_shards = {name for batch in capped for name in batch.source_file_basename}
    assert drawn_shards == set(keys), "a prefix draw would reach one shard"


def test_an_absent_or_slack_sample_cap_leaves_the_loader_untouched(multi_class_loader):
    """The cap is opt-in, and a cap at or above the split size is not a draw at all -- reporting
    one would make two runs of the same checkpoint differ over a setting that changed nothing."""
    from teb_vae.lag_attn_rws.eval.run import capped_sample_loader

    n_total = len(multi_class_loader.dataset)

    same, record = capped_sample_loader(multi_class_loader, None, seed=0)
    slack, slack_record = capped_sample_loader(multi_class_loader, n_total, seed=0)

    assert same is multi_class_loader and slack is multi_class_loader
    assert record["applied"] is False and slack_record["applied"] is False
    assert record["n_drawn"] == n_total
