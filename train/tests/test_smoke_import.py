"""Smoke test: the framework modules and shared helpers import cleanly."""


def test_import_base_modules():
    import train.graph_model_base  # noqa: F401
    import train.pl_model_base  # noqa: F401
    import train.callbacks  # noqa: F401


def test_import_shared_helpers():
    from train.test_utils import (  # noqa: F401
        FakeMLflowLogger,
        FakeTrainer,
        StandInConsumer,
        TinyLightningModel,
        TinyModule,
        make_graph_model,
    )

    assert TinyModule is not None
