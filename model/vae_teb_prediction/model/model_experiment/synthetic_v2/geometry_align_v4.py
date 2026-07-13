r"""S0-T03: the grid-alignment contract between the synthetic DGP and the raw model.

Two grids must be reconciled to grade $\bar K$ against the planted lag $D$ without silently
mis-scoring anything:

* the **synthetic** latents (:func:`generate_cell_raw`) live on the *uncropped* $\tilde T = 330$
  decimated grid; ``true_lag_tt`` is indexed there;
* the model attention (``attn_weights`` / ``te_lag_map``) lives on the *cropped* $T = 300$ grid,
  where cropped anchor $t$ equals uncropped token $t + \mathrm{CROP}$ with $\mathrm{CROP}=15$.

Because the crop is a **pure shift** of the anchor origin and the lag axis is a **relative**
offset, the planted lag maps to the model lag index unchanged; only comparisons that index
*anchors* (event-triggered $K_t$, per-cell profiles, the deferred CMI) must apply the $\pm 15$
offset. :func:`assert_alignment` derives and asserts these identities against the reused
``model_raw`` geometry so a mismatch fails loudly at import/stage entry rather than corrupting a
lag-recovery score.
"""

from __future__ import annotations

from typing import Any, Optional

from model.vae_teb_prediction.model.model_experiment.synthetic_v2.reuse_v4 import geometry

#: The one-sided crop (tokens trimmed each side) between the uncropped 330 and cropped 300 grids.
CROP: int = geometry.CROP


def latent_to_model_step(t: int, *, crop: int = CROP) -> int:
    r"""Map an uncropped latent-grid step $t$ to the cropped model-grid step $t - \mathrm{CROP}$."""
    return t - crop


def model_to_latent_step(t: int, *, crop: int = CROP) -> int:
    r"""Map a cropped model-grid step $t$ to the uncropped latent-grid step $t + \mathrm{CROP}$."""
    return t + crop


def planted_lag_to_model_lag(D: int) -> int:
    r"""Map a planted lag $D$ to the model lag index -- **unchanged**.

    The lag axis is a *relative* past offset (source lag $\ell$ = anchor $-$ source step), and the
    crop shifts both the anchor and the source step by the same $\mathrm{CROP}$, so the relative
    offset is invariant. Hence $\operatorname{planted\_lag\_to\_model\_lag}(D) = D$; this identity
    is what lets lag recovery compare $\operatorname{argmax}_\ell \bar\alpha_{t,\ell}$ directly to
    the planted $D$ with no rescaling.
    """
    return D


def assert_alignment(geom: Optional[Any] = None) -> bool:
    r"""Assert the crop-offset and planted-lag identities against the ``model_raw`` geometry.

    Args:
        geom: A geometry object exposing ``crop`` and ``t_valid`` (defaults to the reused
            production ``geometry.GEOMETRY``). A geometry whose ``crop`` is not $15$ -- i.e. a grid
            that is not a pure $\pm 15$ shift of the model grid -- fails here.

    Returns:
        ``True`` when every identity holds.

    Raises:
        AssertionError: On any violated identity (wrong crop, wrong future-block start, wrong valid
            anchor range, wrong ``t_valid``, or a broken step round-trip).
    """
    g = geom if geom is not None else geometry.GEOMETRY

    crop = int(getattr(g, "crop", CROP))
    if crop != 15:
        raise AssertionError(f"grid alignment requires CROP==15 (pure shift), got crop={crop}")

    # Canonical model-geometry identities (reused, not re-derived).
    assert geometry.CROP == 15, f"geometry.CROP={geometry.CROP} != 15"
    assert geometry.n_raw(0) == 255, f"n_raw(0)={geometry.n_raw(0)} != 255"
    assert geometry.future_block_start(0) == 256, (
        f"future_block_start(0)={geometry.future_block_start(0)} != 256"
    )
    assert geometry.valid_anchor_range() == range(30, 270), (
        f"valid_anchor_range()={geometry.valid_anchor_range()} != range(30, 270)"
    )
    assert int(getattr(g, "t_valid")) == 270, f"t_valid={g.t_valid} != 270"

    # Step round-trip and relative-lag invariance.
    assert latent_to_model_step(geometry.CROP) == 0
    assert model_to_latent_step(0) == geometry.CROP
    for t in (0, 15, 100, 269):
        assert model_to_latent_step(latent_to_model_step(t)) == t
    for D in (0, 1, 8, 90):
        assert planted_lag_to_model_lag(D) == D

    return True
