"""``validate_config`` fails fast on required/mistyped keys, warns on the rest."""
import pytest

from train.test_utils import make_graph_model


def test_shipped_config_passes(config_path):
    gm = make_graph_model(config_path)
    gm.validate_config()  # must not raise


@pytest.mark.parametrize(
    "key, value",
    [
        # A leftover (removed) memory block must load with a warning, never a crash:
        # unmigrated consumer configs still carry it and are out of scope.
        ("advanced_config.memory", {"enable_memory_monitoring": False}),
        ("advanced_config.trainer.made_up_key", 1),
        # 'all' is a permitted string for log_checkpoints.
        ("advanced_config.tracking.mlflow.log_checkpoints", "all"),
    ],
    ids=["legacy-memory-block", "unknown-key", "log_checkpoints-all"],
)
def test_a_tolerated_key_does_not_raise(config_path, key, value):
    gm = make_graph_model(config_path, **{key: value})
    gm.validate_config()  # must not raise


def test_missing_required_key_raises(config_path):
    gm = make_graph_model(config_path)
    del gm.config["general_config"]["cuda_devices"]
    with pytest.raises(ValueError, match="cuda_devices"):
        gm.validate_config()


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("advanced_config.trainer.precision", 16, "precision"),
        ("advanced_config.trainer.compile", "yes", "compile"),
        ("advanced_config.spike_breaker.multiplier", "big", "multiplier"),
        ("advanced_config.tracking.mlflow.enabled", "yes", "enabled"),
        # A non-bool/non-str value must raise a clean ValueError naming the key, not an
        # AttributeError from rendering the (bool, str) tuple in the message.
        ("advanced_config.tracking.mlflow.log_checkpoints", 5, "log_checkpoints"),
    ],
    ids=["precision", "bool", "spike-multiplier", "mlflow-enabled", "log_checkpoints"],
)
def test_a_mistyped_key_raises_naming_it(config_path, key, value, match):
    gm = make_graph_model(config_path, **{key: value})
    with pytest.raises(ValueError, match=match):
        gm.validate_config()


def test_missing_spike_breaker_block_does_not_raise(config_path):
    gm = make_graph_model(config_path)
    del gm.config["advanced_config"]["spike_breaker"]
    gm.validate_config()  # absent block -> breaker OFF, must not raise
