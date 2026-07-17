"""Optional MLflow integration smoke test.

Runs only when ``MLFLOW_SMOKE_URI`` points at a live MLflow server (a manual prod-box
check); otherwise it is skipped. It drives the real framework path — ``setup_config``
(config/hparams/provenance + system-metrics monitor) then a 1-epoch ``Trainer.fit`` with
the ``MLflowRunLoggingCallback`` attached — and asserts the finished run exposes the
model, the resolved config, the architecture record, the parameter counts, and a
non-empty system-metrics series.
"""
import os
import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from train.test_utils import TinyLightningModel, make_graph_model

_SMOKE_URI = os.environ.get("MLFLOW_SMOKE_URI")

pytestmark = pytest.mark.skipif(
    not _SMOKE_URI,
    reason="set MLFLOW_SMOKE_URI to a live MLflow tracking server to run this test",
)


def _tiny_loader():
    x = torch.randn(16, 4)
    y = torch.randn(16, 4)
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_enabled_run_records_model_config_and_system_metrics(config_path, tmp_path):
    import mlflow

    gm = make_graph_model(
        config_path,
        **{
            "general_config.epochs": 1,
            "general_config.cuda_devices": [0],
            "general_config.folders_config.out_dir_base": str(tmp_path),
            "advanced_config.trainer.profiler": None,
            "advanced_config.tracking.mlflow.enabled": True,
            "advanced_config.tracking.mlflow.tracking_uri": _SMOKE_URI,
            "advanced_config.tracking.mlflow.log_model": True,
        },
    )
    gm.setup_config()
    assert gm.mlflow_logger is not None, "MLflow logger must initialise against the server"
    run_id = gm.mlflow_logger.run_id

    model = TinyLightningModel(compile_model=False)
    trainer = gm.build_trainer([], model=model)
    trainer.fit(model, _tiny_loader(), _tiny_loader())

    client = mlflow.tracking.MlflowClient(tracking_uri=_SMOKE_URI)

    def _artifact_paths():
        return {a.path for a in client.list_artifacts(run_id)} | {
            a.path for a in client.list_artifacts(run_id, "model")
        } | {a.path for a in client.list_artifacts(run_id, "config")}

    paths = _artifact_paths()
    assert "model" in paths
    assert "config/resolved_config.yaml" in paths
    assert "model/model_architecture.txt" in paths

    run = client.get_run(run_id)
    assert "params_total" in run.data.metrics

    # The logged model reloads via the pytorch flavour.
    reloaded = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    assert reloaded is not None

    # System metrics arrive on the monitor's sampling interval; poll briefly for one.
    deadline = time.time() + 20
    system_keys = set()
    while time.time() < deadline and not system_keys:
        metrics = client.get_run(run_id).data.metrics
        system_keys = {k for k in metrics if k.startswith("system/")}
        if not system_keys:
            time.sleep(2)
    assert system_keys, "expected a non-empty system-metrics series"
