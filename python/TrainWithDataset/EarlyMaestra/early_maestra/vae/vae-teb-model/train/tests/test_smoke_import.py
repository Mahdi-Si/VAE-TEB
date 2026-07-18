"""Smoke test: the framework modules and shared helpers import cleanly."""


def test_import_base_modules():
    import train.graph_model_base  # noqa: F401
    import train.pl_model_base  # noqa: F401
    import train.callbacks  # noqa: F401
    import train.data_module  # noqa: F401


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


def test_seqvae_plot_callbacks_moved_to_utils():
    """The SeqVAE-coupled plotters resolve under ``utils/`` and are gone from ``train/``.

    Guards the relocation in both directions: the import below executes the new
    module's own ``from train.callbacks import log_artifact_to_mlflow``, so a
    circular import or a botched copy fails here; the assertions catch a class left
    behind in the agnostic module or an untrimmed ``__all__``.
    """
    import train.callbacks
    from utils.seqvae_plot_callbacks import (  # noqa: F401
        PlottingAvgPredCallBack,
        PlottingCallBack,
        ReconstructionPlotCallback,
    )

    for name in ("ReconstructionPlotCallback", "PlottingCallBack", "PlottingAvgPredCallBack"):
        assert not hasattr(train.callbacks, name)
        assert name not in train.callbacks.__all__
