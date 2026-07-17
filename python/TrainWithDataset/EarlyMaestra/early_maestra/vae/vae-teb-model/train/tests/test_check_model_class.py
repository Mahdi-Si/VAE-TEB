"""The checkpoint class guard: the only thing that reads the ``model_class`` stamp back.

``LightningModelBase.on_save_checkpoint`` writes ``model_class`` into every checkpoint, and
``check_model_class`` is what makes that write worth anything. Without it a blob written by one
model loads into another and only fails later, at construction, with a ``TypeError`` naming a
keyword argument rather than the real problem -- or, when the constructors happen to agree, does
not fail at all and silently trains the wrong architecture.

The absent-stamp case warns rather than raises on purpose: checkpoints predating the stamp are
legitimate, and refusing them would make the guard's introduction a breaking change.
"""
import warnings

import pytest

from train.graph_models_utils import check_model_class


def test_a_matching_class_passes_silently():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning at all fails this test
        check_model_class({"model_class": "TinyModule"}, "TinyModule")


def test_a_mismatched_class_raises_and_names_both_classes():
    with pytest.raises(ValueError) as excinfo:
        check_model_class({"model_class": "SeqVaeLagAttnV1"}, "SeqVaeLagAttn")

    message = str(excinfo.value)
    assert "SeqVaeLagAttnV1" in message and "SeqVaeLagAttn" in message


def test_a_missing_model_class_warns_but_does_not_raise():
    """Pre-stamp checkpoints are legitimate; the rebuild still fails loudly on bad kwargs."""
    with pytest.warns(RuntimeWarning, match="no 'model_class' field"):
        check_model_class({"state_dict": {}}, "TinyModule")


def test_a_non_dict_checkpoint_is_skipped():
    """A bare state-dict carries no claim to check, so there is nothing to disagree with."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_model_class(["not", "a", "dict"], "TinyModule")
        check_model_class(None, "TinyModule")


def test_the_comparison_is_on_the_string_value():
    """A stamp is a name, not an object; a non-string value must still compare, not crash."""
    check_model_class({"model_class": 42}, "42")
    with pytest.raises(ValueError):
        check_model_class({"model_class": 42}, "TinyModule")


def test_the_guard_reads_the_key_the_framework_actually_stamps():
    """Pins the two halves together.

    ``on_save_checkpoint`` writing ``model_class`` and this guard reading ``model_class`` are
    the same contract in two files. If either renamed the key unilaterally, every guard call in
    the repo would silently degrade to the warn-and-continue path.
    """
    from train.test_utils import TinyLightningModel

    module = TinyLightningModel()
    checkpoint = {}
    module.on_save_checkpoint(checkpoint)

    assert "model_class" in checkpoint
    check_model_class(checkpoint, checkpoint["model_class"])
