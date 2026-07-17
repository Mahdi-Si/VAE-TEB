"""``validate_config`` fails fast on required/mistyped keys, warns on the rest."""
import pytest

from train.test_utils import make_graph_model


def test_shipped_config_passes(config_path):
    gm = make_graph_model(config_path)
    gm.validate_config()  # must not raise


def test_legacy_memory_block_warns_not_raises(config_path):
    # A leftover (removed) memory block must load with a warning, never a crash —
    # ~27 unmigrated consumer configs still carry it and are out of scope.
    gm = make_graph_model(
        config_path,
        **{"advanced_config.memory": {"enable_memory_monitoring": False}},
    )
    gm.validate_config()  # must not raise


def test_unknown_key_warns_not_raises(config_path):
    gm = make_graph_model(config_path, **{"advanced_config.trainer.made_up_key": 1})
    gm.validate_config()  # must not raise


def test_missing_required_key_raises(config_path):
    gm = make_graph_model(config_path)
    del gm.config["general_config"]["cuda_devices"]
    with pytest.raises(ValueError, match="cuda_devices"):
        gm.validate_config()


def test_mistyped_precision_raises(config_path):
    gm = make_graph_model(config_path, **{"advanced_config.trainer.precision": 16})
    with pytest.raises(ValueError, match="precision"):
        gm.validate_config()


def test_mistyped_bool_raises(config_path):
    gm = make_graph_model(config_path, **{"advanced_config.trainer.compile": "yes"})
    with pytest.raises(ValueError, match="compile"):
        gm.validate_config()


# --- spike_breaker block ------------------------------------------------------

def test_spike_breaker_bad_multiplier_type_raises(config_path):
    gm = make_graph_model(config_path, **{"advanced_config.spike_breaker.multiplier": "big"})
    with pytest.raises(ValueError, match="multiplier"):
        gm.validate_config()


def test_missing_spike_breaker_block_does_not_raise(config_path):
    gm = make_graph_model(config_path)
    del gm.config["advanced_config"]["spike_breaker"]
    gm.validate_config()  # absent block -> breaker OFF, must not raise


# --- tracking.mlflow block ----------------------------------------------------

def test_tracking_bad_enabled_type_raises(config_path):
    gm = make_graph_model(config_path, **{"advanced_config.tracking.mlflow.enabled": "yes"})
    with pytest.raises(ValueError, match="enabled"):
        gm.validate_config()


def test_tracking_log_checkpoints_accepts_all_string(config_path):
    gm = make_graph_model(config_path, **{"advanced_config.tracking.mlflow.log_checkpoints": "all"})
    gm.validate_config()  # 'all' is a permitted string, must not raise


def test_tracking_log_checkpoints_bad_type_raises_valueerror(config_path):
    # A non-bool/non-str value must raise a clean ValueError naming the key, not an
    # AttributeError from rendering the (bool, str) tuple in the message.
    gm = make_graph_model(config_path, **{"advanced_config.tracking.mlflow.log_checkpoints": 5})
    with pytest.raises(ValueError, match="log_checkpoints"):
        gm.validate_config()
