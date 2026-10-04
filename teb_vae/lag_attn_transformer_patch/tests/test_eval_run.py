"""The patch eval end to end: a 2-epoch ``planted.yaml`` fit, then the eval CLI's ``main`` on it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from teb_vae.lag_attn_cfs.lag_recovery_check import write_override_config
from teb_vae.lag_attn_rws.trainer import main as run_training
from teb_vae.lag_attn_transformer_patch.eval import run as run_module
from teb_vae.lag_attn_transformer_patch.eval.binding import NEW_ANALYSES
from teb_vae.lag_attn_transformer_patch.trainer import LagAttnTrfPatchTrainer

from .conftest import _REPO_ROOT

PACKAGE = Path(_REPO_ROOT) / "teb_vae" / "lag_attn_transformer_patch"

#: ``planted_overrides.yaml``'s caps shrunk to a plumbing run (the full instrument takes minutes).
SMALL_CAPS = dict(
    waveforms=4, attention=4, pages=1, pages_per_class=1, traces_per_class=2,
    occlusion=4, time_shift=4,
    raw_attribution_segments=2, raw_attribution_anchors=1, raw_attribution_ig_steps=16,
    fhr_drivers_segments=2, fhr_drivers_anchors=1, fhr_drivers_ig_steps=16,
    impulse_response_segments=2, impulse_response_injections=2, raw_shift_segments=2,
    delay_map_segments=2, delay_map_anchors=2,
    event_locked_segments=4, event_locked_ig_events=1, event_locked_ig_offsets=2, event_locked_ig_steps=16,
    decelerations_segments=4, signal_loss_segments=8, signal_loss_inject_segments=2,
)


@pytest.mark.slow
def test_the_eval_runs_end_to_end_on_a_planted_checkpoint(tmp_path, monkeypatch) -> None:
    """Exit 0 and a ``summary.json``; every step ok, every recorded skip states a reason; each new
    analysis wrote at least one non-empty file."""
    monkeypatch.chdir(_REPO_ROOT)  # the configs' shard paths are repo-root relative
    config = write_override_config(
        str(PACKAGE / "configs" / "planted.yaml"),
        ["general_config.epochs=2", f"general_config.folders_config.out_dir_base={tmp_path / 'train'}"],
        tmp_path,
    )
    run_training(str(config), trainer_cls=LagAttnTrfPatchTrainer)
    checkpoint = sorted((tmp_path / "train").rglob("lag-attn-trf-patch-epoch=01.ckpt"))[0]

    overrides = yaml.safe_load((PACKAGE / "eval" / "configs" / "planted_overrides.yaml").read_text())
    overrides["eval_config"]["caps"].update(SMALL_CAPS)
    overrides_path = tmp_path / "planted_overrides_small.yaml"
    overrides_path.write_text(yaml.safe_dump(overrides, sort_keys=False))

    output_dir = tmp_path / "eval"
    # The shared ``attribution`` (reused unchanged, covered by the CFS suites) is skipped: its IG cost
    # is fixed by the shared ``IG_STEPS`` and no cap, and is ~1 min here.
    exit_code = run_module.main(
        checkpoint, output_dir, overrides=overrides_path, device="cpu", num_samples=2, skip="attribution"
    )

    results_dir = output_dir / run_module.RESULTS_DIRNAME
    summary = json.loads((results_dir / run_module.SUMMARY_FILENAME).read_text())
    failed = {step["name"]: step.get("error") for step in summary["steps"] if step["status"] != "ok"}
    assert failed == {} and exit_code == 0, failed
    results = summary["results"]
    unexplained = [
        name for name, block in results.items()
        if isinstance(block, dict) and block.get("skipped") and not block.get("reason")
    ]
    assert unexplained == [], unexplained
    for module in NEW_ANALYSES:
        name = module.__name__.rsplit(".", 1)[-1]
        written = [path for path in (results_dir / name).rglob("*") if path.is_file() and path.stat().st_size]
        assert written, f"{name} wrote no non-empty file; its block: {results.get(name)}"
