r"""S4-T02 tests: the leaky (non-causal) front end negative control + causality probe."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.v4


def _small_frontend_kwargs():
    r"""The small_prod front-end kwargs (minus the decoder-head keys), for a standalone front end."""
    from model.vae_teb_prediction.model.model_raw.testing.conftest import SMALL_PROD_FRONTEND

    fe = dict(SMALL_PROD_FRONTEND)
    fe.pop("decoder_head", None)
    fe.pop("basis_size", None)
    return fe


def test_causal_frontend_passes_guard_and_probe():
    r"""A `CausalRawFrontend` passes `assert_no_time_pooling_norm` and reads as causal."""
    from model.vae_teb_prediction.model.model_raw.raw_frontend import (
        CausalRawFrontend,
        assert_no_time_pooling_norm,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.leaky_frontend_v4 import (
        frontend_is_causal,
    )

    cf = CausalRawFrontend(
        stream="y", mean=0.0, std=1.0, raw_len=5280, decimation=16, sentinel=None,
        **_small_frontend_kwargs(),
    )
    assert_no_time_pooling_norm(cf)  # must not raise
    assert frontend_is_causal(cf) is True


def test_leaky_frontend_fails_guard_and_probe():
    r"""A `LeakyRawFrontend` fails the time-pooling guard and reads as non-causal."""
    from model.vae_teb_prediction.model.model_raw.raw_frontend import assert_no_time_pooling_norm
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.leaky_frontend_v4 import (
        LeakyRawFrontend,
        frontend_is_causal,
    )

    lf = LeakyRawFrontend(
        stream="y", mean=0.0, std=1.0, raw_len=5280, decimation=16, sentinel=None,
        **_small_frontend_kwargs(),
    )
    with pytest.raises(ValueError):
        assert_no_time_pooling_norm(lf)
    assert frontend_is_causal(lf) is False


def test_leaky_model_builds_and_frontends_are_leaky():
    r"""`LeakyRawFrontendSeqVaeRawV4` builds; its front ends fail the guard, a plain model passes."""
    from model.vae_teb_prediction.model.model_raw.raw_frontend import assert_no_time_pooling_norm
    from model.vae_teb_prediction.model.model_raw.testing.conftest import (
        SMALL_PROD_FRONTEND,
        SMALL_PROD_V3_KWARGS,
        make_small_prod_raw_model,
    )
    from model.vae_teb_prediction.model.model_experiment.synthetic_v2.leaky_frontend_v4 import (
        LeakyRawFrontend,
        LeakyRawFrontendSeqVaeRawV4,
        frontend_is_causal,
    )

    leaky_model = LeakyRawFrontendSeqVaeRawV4(
        frontend=dict(SMALL_PROD_FRONTEND), raw_len=5280, decimation=16, **SMALL_PROD_V3_KWARGS,
    )
    assert isinstance(leaky_model.frontend_y, LeakyRawFrontend)
    assert isinstance(leaky_model.frontend_u, LeakyRawFrontend)
    with pytest.raises(ValueError):
        assert_no_time_pooling_norm(leaky_model.frontend_y)
    assert frontend_is_causal(leaky_model.frontend_y) is False

    prod_model = make_small_prod_raw_model()
    assert_no_time_pooling_norm(prod_model.frontend_y)  # must not raise
    assert frontend_is_causal(prod_model.frontend_y) is True
