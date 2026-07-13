r"""S5-T05: single-GPU training-hardening smoke for synthetic_v4 (memory / bf16 / resume).

Three checks that de-risk the headline run before the 8x A6000 DDP sweep:

* **memory sizing** -- one ``prod`` forward+backward at the full raw length ($L=5280$) records the
  peak allocated GPU memory, so the headline ``batch_size`` can be sized against the front end's
  $16\times$-longer sequences (GPU-gated);
* **bf16-mixed** -- one autocast ``bf16`` step produces a finite loss (GPU-gated);
* **resume** -- a checkpoint saved mid-run resumes and *continues* from the saved ``global_step``
  (device-independent Lightning bookkeeping; runs on CPU).

The whole module is marked ``slow``; the two GPU checks ``skipif`` when CUDA is absent. The single
documented smoke command for the prod box is::

    pytest .../tests/test_hardening_v4.py -m slow
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[6])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

import pytest  # noqa: E402

pytestmark = [pytest.mark.v4, pytest.mark.slow]

_CUDA = None


def _has_cuda() -> bool:
    global _CUDA
    if _CUDA is None:
        import torch

        _CUDA = bool(torch.cuda.is_available())
    return _CUDA


def _small_prod_model(**overrides):
    from model.vae_teb_prediction.model.model_raw.testing.conftest import make_small_prod_raw_model

    return make_small_prod_raw_model(**overrides)


def _small_prod_batch(batch_size: int, *, device):
    r"""A full-geometry ($L=5280$) raw batch on ``device`` (``fhr`` / ``up`` / decimated ``weight``)."""
    import torch

    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        SMALL_PROD_DECIMATION,
        SMALL_PROD_RAW_LEN,
    )

    g = torch.Generator().manual_seed(0)
    t_tilde = SMALL_PROD_RAW_LEN // SMALL_PROD_DECIMATION
    fhr = torch.randn(batch_size, SMALL_PROD_RAW_LEN, generator=g).to(device)
    up = torch.randn(batch_size, SMALL_PROD_RAW_LEN, generator=g).to(device)
    mask = torch.ones(batch_size, SMALL_PROD_RAW_LEN, device=device)
    weight = torch.ones(batch_size, t_tilde, device=device)
    return fhr, up, mask, weight


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for memory sizing")
def test_full_length_peak_memory_recorded() -> None:
    r"""One ``prod`` forward+backward at $L=5280$ records the peak allocated GPU memory."""
    import torch

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    model = _small_prod_model().to(device).train()
    # A modest batch keeps the RTX 4080 smoke in-budget; the prod box scales this up.
    fhr, up, mask, _ = _small_prod_batch(batch_size=4, device=device)

    out = model(fhr, up, mask)
    loss = model.compute_loss(out, fhr, mask, beta=0.1, free_bits=0.0)["total_loss"]
    loss.backward()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"[hardening] prod full-length (L=5280, B=4) peak GPU memory: {peak_mb:.1f} MiB")
    assert torch.isfinite(loss).all()
    assert peak_mb > 0.0


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for bf16-mixed")
def test_bf16_mixed_step_finite() -> None:
    r"""One ``bf16``-autocast forward+backward yields a finite loss."""
    import torch

    device = torch.device("cuda")
    model = _small_prod_model().to(device).train()
    fhr, up, mask, _ = _small_prod_batch(batch_size=2, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(fhr, up, mask)
        loss = model.compute_loss(out, fhr, mask, beta=0.1, free_bits=0.0)["total_loss"]
    loss.backward()
    assert torch.isfinite(loss).all()


def test_resume_continues_from_saved_step(tmp_path) -> None:
    r"""A checkpoint saved after 2 steps resumes and continues to step 4 (Lightning bookkeeping).

    Device-independent: run on CPU with the tiny model + a small in-memory loader so the check is
    about ``global_step`` continuity, not GPU throughput.
    """
    import lightning as pl
    import torch
    from torch.utils.data import DataLoader, Dataset

    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        TINY_DECIMATION,
        TINY_RAW_LEN,
        RawStubBatch,
        make_tiny_raw_model,
        tiny_raw_kwargs,
    )
    from model.vae_teb_prediction.model.model_raw.trainer_raw_v4 import SeqVaeRawV4Pl

    t_tilde = TINY_RAW_LEN // TINY_DECIMATION

    class _RawDS(Dataset):
        def __len__(self) -> int:
            return 8

        def __getitem__(self, i: int):
            g = torch.Generator().manual_seed(i)
            return (torch.randn(TINY_RAW_LEN, generator=g),
                    torch.randn(TINY_RAW_LEN, generator=g),
                    torch.ones(t_tilde))

    def _collate(items):
        return RawStubBatch(
            fhr=torch.stack([it[0] for it in items]),
            up=torch.stack([it[1] for it in items]),
            weight=torch.stack([it[2] for it in items]),
            guid=[f"stub{k:04d}" for k in range(len(items))],
        )

    def _loader():
        # A real DataLoader (not a bare list, which Lightning reads as a list-of-dataloaders).
        return DataLoader(_RawDS(), batch_size=2, collate_fn=_collate)

    def _trainer(max_steps: int):
        return pl.Trainer(
            max_steps=max_steps, accelerator="cpu", devices=1, logger=False,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            num_sanity_val_steps=0,
        )

    kwargs = tiny_raw_kwargs()
    pl_a = SeqVaeRawV4Pl(make_tiny_raw_model(), lr=1e-3, model_kwargs=kwargs)
    t1 = _trainer(2)
    t1.fit(pl_a, train_dataloaders=_loader())
    assert t1.global_step == 2

    ckpt = tmp_path / "resume.ckpt"
    t1.save_checkpoint(str(ckpt))
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    assert int(blob["global_step"]) == 2

    pl_b = SeqVaeRawV4Pl(make_tiny_raw_model(), lr=1e-3, model_kwargs=kwargs)
    t2 = _trainer(4)
    t2.fit(pl_b, train_dataloaders=_loader(), ckpt_path=str(ckpt))
    assert t2.global_step == 4, "resumed run did not continue from the saved step"
