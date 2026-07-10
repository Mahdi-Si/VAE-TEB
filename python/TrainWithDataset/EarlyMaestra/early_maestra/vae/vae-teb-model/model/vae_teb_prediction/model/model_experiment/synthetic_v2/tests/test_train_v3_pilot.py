r"""S2-T05: pilot beta schedule and the bottleneck-health gate.

v3 starts at $K \equiv 0$ and must GROW $K$ to earn prediction gain. The failure this guards
against is a silent null result: the source path never switches on, ``kld_raw`` and
``pred_gap`` both sit at zero, and ``total_loss`` still looks healthy. The gate is therefore
two-sided and thresholded on NAMED constants rather than on ``> 0`` (float noise makes
``1e-9 > 0`` a false pass on a collapsed run):

* step-0 ``train/kld_raw`` is ``< 1e-6`` for the residual-posterior v3 arms and ``> 1e-3``
  for ``parity`` (whose independent log-variance head carries a random KL floor);
* end-of-pilot ``median(train/kld_raw) > EPS_OPEN`` (the bottleneck opened) and
  ``median(train/pred_gap) > EPS_GAP`` (the source is USEFUL), over three seeds so one
  unlucky init cannot fail CI;
* no NaN/Inf, ``mean_logvar_full`` stays above the ``-5`` floor;
* every arm writes ``{final,best}.ckpt`` under ``results/<tag>/<arm>/`` carrying
  ``model_class`` / resolved ``model_kwargs`` / ``arm``.

Thresholds (measured; see the ``train:`` block of ``config_synth_v3.yaml``): at the shipped
400-step pilot the three seeds gave ``kld_raw`` median 0.0376 (min 0.0197) and ``pred_gap``
median +4.9e-3 (min +3.6e-3). ``EPS_OPEN``/``EPS_GAP`` sit ~20x/~36x below the worst seed.

Slow: trains all three arms on the real cache. Skipped when no cache exists.
"""

from __future__ import annotations

import csv
import statistics
import sys
import warnings
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import math  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.pl_module_v2 import (  # noqa: E402,E501
    train_v2,
)
from model.vae_teb_prediction.model.model_experiment.synthetic_v2.run_pipeline_v2 import (  # noqa: E402,E501
    load_config,
    resolve_arm,
)

pytestmark = pytest.mark.slow

_SV2 = Path(__file__).resolve().parents[1]

#: The bottleneck must be OPEN: the raw per-step KL is materially above float noise.
EPS_OPEN = 1.0e-3
#: The source must be USEFUL: the full forecast beats the target-only baseline.
EPS_GAP = 1.0e-4
#: Learned observation log-variance must not collapse onto the smooth lower bound.
LOGVAR_FLOOR = -5.0

_ARMS = ("parity", "v3_noncausal", "v3_prod")
_SEEDS = (0, 1, 2)


def _pilot_config(tag: str):
    cfg = load_config(_SV2 / "config_synth_v3.yaml")
    cfg["experiment"]["tag"] = tag
    cfg["plotting"] = {"enabled": False, "plot_every": 999, "html": False}
    cfg["dataset"] = {"num_workers": 0, "pin_memory": False,
                      "persistent_workers": False, "mmap": "auto"}
    return cfg


def _pilot_overrides(cfg, *, arm=None, skip_checkpoint=False):
    t = cfg["train"]
    ov = {
        "pilot": True,
        "epochs": int(t["pilot_epochs"]),
        "limit_train_batches": int(t["pilot_limit_train_batches"]),
        "limit_val_batches": int(t["pilot_limit_val_batches"]),
        "batch_size": int(t["pilot_batch_size"]),
        "devices": 1,
        "progress_bar": False,
        "skip_checkpoint": skip_checkpoint,
    }
    if arm is not None:
        ov["arm"] = arm
    return ov


def _step0_kld_raw(metrics_csv: Path):
    for row in csv.DictReader(metrics_csv.open()):
        if row.get("train/kld_raw_step"):
            return float(row["train/kld_raw_step"])
    return None


def _require_cache(shared_cache_dir):
    if shared_cache_dir is None:
        pytest.skip("no cache built yet (S0-T00); the pilot gate needs real features")


def test_pilot_beta_schedule_is_configured() -> None:
    """``train.pilot_beta_schedule`` exists and is a SHORT warm-up (unlike the headline ramp)."""
    cfg = load_config(_SV2 / "config_synth_v3.yaml")
    pilot = cfg["train"]["pilot_beta_schedule"]
    headline = cfg["loss"]["beta_schedule"]
    assert pilot["kind"] == "linear_warmup"
    assert int(pilot["warmup_epochs"]) < int(headline["warmup_epochs"])
    assert int(pilot["warmup_epochs"]) <= int(cfg["train"]["pilot_epochs"]), (
        "the pilot warm-up must complete inside the pilot budget, else the gate cannot fire"
    )


def test_bottleneck_health_gate_median_over_seeds(shared_cache_dir) -> None:
    """Over 3 seeds, the median end-of-pilot kld_raw and pred_gap clear their thresholds."""
    _require_cache(shared_cache_dir)
    kld_raw, pred_gap = [], []
    for seed in _SEEDS:
        cfg = resolve_arm(_pilot_config("G1_raw_v3_pilottest"), "v3_prod")
        cfg["seeds"] = {**cfg["seeds"], "base_seed": seed}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = train_v2(cfg, _pilot_overrides(cfg, skip_checkpoint=True),
                               benchmark="G1_raw")["metrics"]
        kld_raw.append(metrics["train/kld_raw_epoch"])
        pred_gap.append(metrics["train/pred_gap_epoch"])
        assert math.isfinite(metrics["train/total_loss_epoch"]), f"seed {seed}: non-finite loss"

    med_kld = statistics.median(kld_raw)
    med_gap = statistics.median(pred_gap)
    assert med_kld > EPS_OPEN, (
        f"bottleneck did not open: median kld_raw={med_kld:.3e} <= EPS_OPEN={EPS_OPEN:.0e} "
        f"(per-seed {kld_raw})"
    )
    assert med_gap > EPS_GAP, (
        f"source path not useful: median pred_gap={med_gap:.3e} <= EPS_GAP={EPS_GAP:.0e} "
        f"(per-seed {pred_gap})"
    )


@pytest.mark.parametrize("arm", _ARMS)
def test_pilot_arm_completes_with_provenance(arm, shared_cache_dir) -> None:
    """Each arm pilot-trains, writes {final,best}.ckpt under results/<tag>/<arm>/, and its
    step-0 kld_raw matches the zero-KL-init contract."""
    _require_cache(shared_cache_dir)
    cfg = resolve_arm(_pilot_config("G1_raw_v3_pilottest"), arm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = train_v2(cfg, _pilot_overrides(cfg, arm=arm), benchmark="G1_raw")

    final_path = Path(result["checkpoint"])
    best_path = Path(result["best"])
    assert final_path.is_file() and best_path.is_file()
    assert final_path.parent.name == arm, "checkpoints must be arm-scoped"

    # Step-0 zero-KL contract on a real training step.
    step0 = _step0_kld_raw(Path(result["metrics_csv"]))
    assert step0 is not None
    if arm == "parity":
        assert step0 > 1e-3, f"[parity] step-0 kld_raw={step0:.3e} (expected > 1e-3)"
    else:
        assert step0 < 1e-6, f"[{arm}] step-0 kld_raw={step0:.3e} (expected < 1e-6)"

    metrics = result["metrics"]
    assert math.isfinite(metrics["train/total_loss_epoch"])
    assert metrics["train/mean_logvar_full_epoch"] > LOGVAR_FLOOR, "variance collapse"

    blob = torch.load(str(final_path), map_location="cpu", weights_only=False)
    assert blob["model_class"] == "SeqVaeLagAttnV3"
    assert blob["arm"] == arm
    expected_causal = arm == "v3_prod"
    assert blob["model_kwargs"]["causal_norm"] is expected_causal
    expected_plogvar = "independent" if arm == "parity" else "residual"
    assert blob["model_kwargs"]["posterior_logvar"] == expected_plogvar
    # The checkpoint records the schedule that ACTUALLY ran (the pilot ramp, not the headline).
    assert blob["loss_settings"]["beta_schedule"] == cfg["train"]["pilot_beta_schedule"]
