import numpy as np
import pytest
import torch

from model.vae_teb_prediction.testing.TE_Calculated.te_kld_analysis import (
    pca_trajectory,
)
from model.vae_teb_prediction.testing.metrics import (
    compute_kld_aggregate_tensors,
    compute_kld_aggregates_per_sample,
    select_pca_components,
)


def test_kld_mean_sum_l2_aggregates_respect_warmup():
    mu_prior = torch.zeros(1, 4, 2)
    logvar_prior = torch.zeros(1, 4, 2)
    logvar_post = torch.zeros(1, 4, 2)
    mu_post = torch.tensor(
        [[[10.0, 10.0], [1.0, 2.0], [2.0, 0.0], [0.0, 4.0]]]
    )
    outputs = {
        "mu_prior": mu_prior,
        "logvar_prior": logvar_prior,
        "mu_post": mu_post,
        "logvar_post": logvar_post,
    }

    per_t = compute_kld_aggregate_tensors(outputs, warmup_steps=1)
    assert per_t is not None

    # Per-dim KL is 0.5 * delta_mu^2 because variances match.
    expected_dim = torch.tensor([[float("nan"), 1.25, 1.0, 4.0]])
    expected_sum = torch.tensor([[float("nan"), 2.5, 2.0, 8.0]])
    expected_l2 = torch.tensor([[float("nan"), np.sqrt(0.5**2 + 2.0**2), 2.0, 8.0]])

    assert torch.allclose(per_t["kld_mean_t"][:, 1:], expected_dim[:, 1:])
    assert torch.isnan(per_t["kld_sum_t"][0, 0])
    assert torch.isnan(per_t["kld_l2_t"][0, 0])
    assert torch.allclose(per_t["kld_sum_t"][:, 1:], expected_sum[:, 1:])
    assert torch.allclose(per_t["kld_l2_t"][:, 1:], expected_l2[:, 1:])

    per_sample = compute_kld_aggregates_per_sample(outputs, warmup_steps=1)
    assert per_sample["kld_mean"].item() == pytest.approx((1.25 + 1.0 + 4.0) / 3.0)
    assert per_sample["kld_sum"].item() == pytest.approx((2.5 + 2.0 + 8.0) / 3.0)
    assert per_sample["kld_l2"].item() == pytest.approx(
        (np.sqrt(0.5**2 + 2.0**2) + 2.0 + 8.0) / 3.0
    )


def test_select_pca_components_uses_label_contrast_before_eigenvalue_rank():
    projected = np.zeros((6, 2, 3), dtype=float)
    projected[:, :, 0] = np.array([0.1, -0.1, 0.0, 0.1, -0.1, 0.0])[:, None]
    projected[:, :, 1] = np.array([-3, -2, -1, 1, 2, 3], dtype=float)[:, None]
    projected[:, :, 2] = np.linspace(-1, 1, 6)[:, None]
    labels = np.array([1, 1, 1, 3, 3, 3])

    selected = select_pca_components(
        projected,
        explained_variance_ratio=[0.8, 0.2, 0.1],
        n_select=1,
        labels=labels,
    )

    assert selected["contrast_type"] == "label_contrast"
    assert selected["selected_indices"].tolist() == [1]


def test_pca_trajectory_selected_modes_from_component_columns():
    import pandas as pd

    df = pd.DataFrame(
        {
            "kld_pc_selected_1": [3.0, -4.0],
            "kld_pc_selected_2": [4.0, 3.0],
        }
    )

    assert pca_trajectory(df, "l2_selected").tolist() == pytest.approx([5.0, 5.0])
    assert pca_trajectory(df, "abs_sum_selected").tolist() == pytest.approx([7.0, 7.0])
    assert pca_trajectory(df, "signed_sum_selected").tolist() == pytest.approx([7.0, -1.0])
