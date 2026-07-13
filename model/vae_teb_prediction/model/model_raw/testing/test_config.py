r"""S0-T03: `config_raw_v4.yaml` parses and carries the required raw v4 keys.

The authoritative fill (final loss scales, tuned batch, stats-trim assertion) is S4-T03; this
skeleton test only pins the load-bearing contract keys so a later edit cannot silently drop them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_raw_v4.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with _CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_config_parses_and_has_top_level_keys(cfg: dict) -> None:
    for key in ("general_config", "model_config", "dataset_config", "advanced_config"):
        assert key in cfg, f"missing top-level key: {key}"


def test_vae_model_v3_flags(cfg: dict) -> None:
    vae = cfg["model_config"]["VAE_model"]
    assert vae["causal_norm"] is True
    assert vae["logvar_bound"] == "smooth"
    assert vae["posterior_logvar"] == "residual"
    assert vae["kld_support"] == "anchor"
    assert vae["lag_bias_init"] == "alibi_decay"
    assert vae["head_structured_latent"] is True
    assert vae["freeze_unused_attn_proj"] is True
    assert vae["lambda_perm"] == 0.0
    assert vae["likelihood"] == "gaussian_nll"
    assert vae["sigma_obs"] == "learned"
    assert vae["free_bits"] == 0.1


def test_raw_geometry_keys(cfg: dict) -> None:
    vae = cfg["model_config"]["VAE_model"]
    assert vae["raw_len"] == 5280
    assert vae["decimation"] == 16
    assert vae["sequence_length"] == 300


def test_frontend_block(cfg: dict) -> None:
    fe = cfg["model_config"]["VAE_model"]["frontend"]
    assert fe["norm_kind"] == "causal_group_norm"
    assert fe["d_raw"] == 128
    assert fe["stages"] == [2, 2, 2, 2]
    assert fe["channels"] == [32, 64, 96, 128]
    assert fe["antialias"] is True
    assert fe["gated"] is True
    assert fe["first_kernels_fhr"] == [7, 31, 65]
    assert fe["first_kernels_up"] == [15, 65, 129]
    # norm_num_groups must divide every stage channel count (CausalGroupNorm requirement).
    g = fe["norm_num_groups"]
    assert all(c % g == 0 for c in fe["channels"])


def test_raw_loss_weight_keys(cfg: dict) -> None:
    vae = cfg["model_config"]["VAE_model"]
    assert "lambda_lp" in vae
    assert "lambda_smooth" in vae
    assert vae["lowpass_scales"] == [4, 16, 32, 60]


def test_dataset_loader_is_untrimmed_raw(cfg: dict) -> None:
    dl = cfg["dataset_config"]["dataloader_config"]
    assert dl["normalize_fields"] == ["fhr", "up"]
    dk = dl["dataset_kwargs"]
    # Untrimmed: full 22-min window (raw_len 5280 / decimated 330).
    assert dk["trim_minutes"] is None
    for field in ("fhr", "up", "weight", "target"):
        assert field in dk["load_fields"]


def test_warm_start_and_checkpoint_siblings(cfg: dict) -> None:
    mc = cfg["model_config"]
    assert mc["warm_start_from"] is None
    assert mc["core_model_checkpoint"] is None
