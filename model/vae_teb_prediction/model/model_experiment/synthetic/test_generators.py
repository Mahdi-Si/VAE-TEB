"""Pytest correctness checks for ``generators`` -- Task 1.5 of the synthetic-TE plan.

Verifies the Benchmark A delayed linear-Gaussian generator: native shapes, the
empirical SNR matching $1 + a^2/\\sigma^2$, distractor channels carrying no
transfer, ground-truth metadata, determinism, standardisation, and argument
validation. Run from the repo root with ``python -m pytest``.
"""

import pytest
import torch

from model.vae_teb_prediction.model.model_experiment.synthetic.analytic_te import (
    te_block_gaussian,
    te_block_xor,
)
from model.vae_teb_prediction.model.model_experiment.synthetic.generators import (
    gen_ar_gaussian,
    gen_delayed_gaussian,
    gen_delayed_xor,
    gen_two_lag_gaussian,
)

N = 2000
T = 300
DELAY = 60
HORIZON = 30


def _lagged_corr(src: torch.Tensor, tgt: torch.Tensor, delay: int) -> float:
    """Pearson correlation between ``src(t - delay)`` and ``tgt(t)``, pooled."""
    x = src[:, : src.shape[1] - delay].reshape(-1)
    y = tgt[:, delay:].reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y) / denom) if denom > 0 else 0.0


def test_native_shapes_and_dtype():
    Y, U, _ = gen_delayed_gaussian(8, T, DELAY, 1.0, 1.0, 4, seed=0)
    assert Y.shape == (8, T, 87)
    assert U.shape == (8, T, 101)
    assert Y.dtype == torch.float32 and U.dtype == torch.float32


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
def test_informative_variance_ratio(a):
    # On the source-driven region t >= delay, Var(Y_j) = a^2 + sigma^2, so
    # Var(Y_j) / sigma^2 = 1 + a^2 / sigma^2.
    sigma2 = 1.0
    Y, _, meta = gen_delayed_gaussian(
        N, T, DELAY, a, sigma2, M=4, standardize=False, seed=0
    )
    for j in meta["informative_channels"]:
        ratio = Y[:, DELAY:, j].var().item() / sigma2
        assert ratio == pytest.approx(1.0 + a * a / sigma2, rel=0.08)


def test_distractor_channels_carry_no_transfer():
    Y, U, meta = gen_delayed_gaussian(
        N, T, DELAY, 1.0, 1.0, M=4, standardize=False, seed=0
    )
    # Informative channel: strong lagged correlation at the true delay.
    assert abs(_lagged_corr(U[:, :, 0], Y[:, :, 0], DELAY)) > 0.5
    # Distractor channels: ~zero lagged correlation.
    for j in range(meta["M"], 87):
        assert abs(_lagged_corr(U[:, :, j], Y[:, :, j], DELAY)) < 0.05


def test_meta_te_true_matches_formula():
    _, _, meta = gen_delayed_gaussian(8, T, DELAY, 1.0, 1.0, M=4, seed=0)
    assert meta["te_true"] == pytest.approx(
        te_block_gaussian(1.0, 1.0, HORIZON, 4), abs=1e-9
    )
    assert meta["true_lag_band"] == list(range(DELAY - HORIZON, DELAY))
    assert meta["clean_anchor_range"] == [DELAY - 1, T - HORIZON]
    assert meta["informative_channels"] == [0, 1, 2, 3]


def test_determinism():
    args = dict(n=8, T=T, delay=DELAY, a=1.0, sigma2=1.0, M=4)
    Y1, U1, _ = gen_delayed_gaussian(**args, seed=0)
    Y2, U2, _ = gen_delayed_gaussian(**args, seed=0)
    assert torch.equal(Y1, Y2) and torch.equal(U1, U2)
    Y3, _, _ = gen_delayed_gaussian(**args, seed=1)
    assert not torch.equal(Y1, Y3)


def test_zero_transfer_null():
    _, _, meta = gen_delayed_gaussian(8, T, DELAY, 0.0, 1.0, M=4, seed=0)
    assert meta["te_true"] == 0.0


def test_standardize_unit_variance():
    Y, U, _ = gen_delayed_gaussian(
        N, T, DELAY, 2.0, 1.0, M=4, standardize=True, seed=0
    )
    for j in range(87):
        assert Y[:, :, j].var().item() == pytest.approx(1.0, rel=0.05)
    for j in range(101):
        assert U[:, :, j].var().item() == pytest.approx(1.0, rel=0.05)


def test_easy_variant_all_channels_informative():
    _, _, meta = gen_delayed_gaussian(
        8, T, DELAY, 1.0, 1.0, M=4, easy_variant=True, seed=0
    )
    assert meta["M"] == 87
    assert meta["informative_channels"] == list(range(87))


def test_delay_less_than_horizon_raises():
    with pytest.raises(ValueError):
        gen_delayed_gaussian(8, T, delay=10, a=1.0, sigma2=1.0, M=4, horizon=30)


def test_invalid_m_raises():
    with pytest.raises(ValueError):
        gen_delayed_gaussian(8, T, DELAY, 1.0, 1.0, M=200, seed=0)


# =============================================================================
# Benchmark G -- reverse-roles directionality variant (task 7.4)
# =============================================================================

def test_g_reverse_roles_zero_te():
    Y, U, meta = gen_delayed_gaussian(
        8, T, DELAY, 1.0, 1.0, M=4, reverse_roles=True, seed=0
    )
    assert Y.shape == (8, T, 87) and U.shape == (8, T, 101)
    assert meta["te_true"] == 0.0
    assert meta["true_lag_band"] == []
    assert meta["benchmark"] == "G"
    assert meta["direction"] == "Y_to_X"
    assert meta["reverse_roles"] is True


def test_g_reverse_roles_swaps_streams():
    # The dependent stream lives in the 101-ch source slot: U_j(t) depends on
    # the i.i.d. target Y_j(t - D). The model-measured direction corr(U(t-D),
    # Y(t)) is ~0 (anti-causal); corr(Y(t-D), U(t)) is strong (the true arrow).
    Y, U, _ = gen_delayed_gaussian(
        N, T, DELAY, 1.0, 1.0, M=4, reverse_roles=True,
        standardize=False, seed=0,
    )
    assert abs(_lagged_corr(Y[:, :, 0], U[:, :, 0], DELAY)) > 0.5
    assert abs(_lagged_corr(U[:, :, 0], Y[:, :, 0], DELAY)) < 0.05


# =============================================================================
# Benchmark B -- AR target + delayed source (task 7.1)
# =============================================================================

def test_b_native_shapes_and_dtype():
    Y, U, _ = gen_ar_gaussian(8, T, DELAY, 0.5, 1.0, 4, rho=0.9, seed=0)
    assert Y.shape == (8, T, 87) and U.shape == (8, T, 101)
    assert Y.dtype == torch.float32 and U.dtype == torch.float32


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9, 0.99])
def test_b_te_true_independent_of_rho(rho):
    # The AR self-term cancels in the determinant ratio -> te_true is the
    # Benchmark-A formula, independent of rho.
    _, _, meta = gen_ar_gaussian(8, T, DELAY, 0.5, 1.0, 4, rho=rho, seed=0)
    assert meta["te_true"] == pytest.approx(
        te_block_gaussian(0.5, 1.0, HORIZON, 4), abs=1e-9
    )
    assert meta["rho"] == pytest.approx(rho)


def test_b_zero_a_null():
    for rho in (0.0, 0.5, 0.9, 0.99):
        _, _, meta = gen_ar_gaussian(8, T, DELAY, 0.0, 1.0, 4, rho=rho, seed=0)
        assert meta["te_true"] == 0.0


def test_b_stationarity_after_burn_in():
    # With the burn-in discarded, the kept window is AR-stationary: the raw
    # informative-channel variance matches (a^2 + sigma^2) / (1 - rho^2).
    a, sigma2, rho = 0.5, 1.0, 0.9
    Y, _, meta = gen_ar_gaussian(
        N, T, DELAY, a, sigma2, 4, rho=rho, standardize=False, seed=0
    )
    expected = (a * a + sigma2) / (1.0 - rho * rho)
    var = Y[:, meta["delay"]:, 0].var().item()
    assert var == pytest.approx(expected, rel=0.12)


def test_b_determinism_and_invalid_rho():
    args = dict(n=8, T=T, delay=DELAY, a=0.5, sigma2=1.0, M=4, rho=0.9)
    Y1, U1, _ = gen_ar_gaussian(**args, seed=0)
    Y2, U2, _ = gen_ar_gaussian(**args, seed=0)
    assert torch.equal(Y1, Y2) and torch.equal(U1, U2)
    with pytest.raises(ValueError):
        gen_ar_gaussian(8, T, DELAY, 0.5, 1.0, 4, rho=1.0, seed=0)


# =============================================================================
# Benchmark C -- delayed binary XOR (task 7.2)
# =============================================================================

def test_c_native_shapes_and_dtype():
    Y, U, _ = gen_delayed_xor(8, T, DELAY, 0.10, 4, seed=0)
    assert Y.shape == (8, T, 87) and U.shape == (8, T, 101)
    assert Y.dtype == torch.float32 and U.dtype == torch.float32


def test_c_te_true_matches_xor_formula():
    _, _, meta = gen_delayed_xor(8, T, DELAY, 0.10, 4, seed=0)
    assert meta["te_true"] == pytest.approx(
        te_block_xor(0.10, HORIZON, 4), abs=1e-9
    )
    assert meta["benchmark"] == "C"
    assert meta["q"] == pytest.approx(0.10)
    # XOR meta carries q / obs_noise, not the Gaussian a / sigma2.
    assert "a" not in meta and "sigma2" not in meta


def test_c_bit_agreement():
    # On raw data, sign(Y_j(t)) agrees with sign(X_j(t-D)) at probability 1-q
    # on informative channels and ~0.5 on distractors.
    q = 0.10
    Y, U, meta = gen_delayed_xor(
        400, T, DELAY, q, 4, obs_noise=0.1, standardize=False, seed=0,
    )
    y_bit = (Y[:, DELAY:, :] > 0).float()
    x_bit = (U[:, : T - DELAY, :87] > 0).float()
    agree = (y_bit == x_bit).float().mean(dim=(0, 1))
    for j in meta["informative_channels"]:
        assert agree[j].item() == pytest.approx(1.0 - q, abs=0.03)
    for j in range(meta["M"], 87):
        assert agree[j].item() == pytest.approx(0.5, abs=0.03)


def test_c_q_half_null():
    _, _, meta = gen_delayed_xor(8, T, DELAY, 0.5, 4, seed=0)
    assert meta["te_true"] == pytest.approx(0.0, abs=1e-9)


# =============================================================================
# Benchmark E -- two-lag Gaussian (task 7.3)
# =============================================================================

def test_e_native_shapes_and_additive_te():
    Y, U, meta = gen_two_lag_gaussian(
        8, T, 50, 80, 0.4, 0.25, 1.0, 4, 4, seed=0
    )
    assert Y.shape == (8, T, 87) and U.shape == (8, T, 101)
    expected = (te_block_gaussian(0.4, 1.0, HORIZON, 4)
                + te_block_gaussian(0.25, 1.0, HORIZON, 4))
    assert meta["te_true"] == pytest.approx(expected, abs=1e-9)
    assert meta["te_true_1"] == pytest.approx(
        te_block_gaussian(0.4, 1.0, HORIZON, 4), abs=1e-9
    )


def test_e_two_lag_bands_in_meta():
    _, _, meta = gen_two_lag_gaussian(
        8, T, 50, 80, 0.4, 0.25, 1.0, 4, 4, seed=0
    )
    assert meta["lag_band_1"] == list(range(20, 50))
    assert meta["lag_band_2"] == list(range(50, 80))
    assert meta["true_lag_band"] == list(range(20, 80))   # deduped union
    assert meta["benchmark"] == "E"


def test_e_group_lag_separation():
    # Group-1 channels transfer at delay1 only, group-2 channels at delay2 only.
    Y, U, meta = gen_two_lag_gaussian(
        N, T, 50, 80, 0.5, 0.5, 1.0, 4, 4, standardize=False, seed=0
    )
    m1 = meta["M1"]
    # Group-1 channel 0: strong at delay1, weak at delay2.
    assert abs(_lagged_corr(U[:, :, 0], Y[:, :, 0], 50)) > 0.3
    assert abs(_lagged_corr(U[:, :, 0], Y[:, :, 0], 80)) < 0.1
    # Group-2 channel M1: strong at delay2, weak at delay1.
    assert abs(_lagged_corr(U[:, :, m1], Y[:, :, m1], 80)) > 0.3
    assert abs(_lagged_corr(U[:, :, m1], Y[:, :, m1], 50)) < 0.1


def test_e_equal_delays_raises():
    with pytest.raises(ValueError):
        gen_two_lag_gaussian(8, T, 50, 50, 0.4, 0.25, 1.0, 4, 4, seed=0)
