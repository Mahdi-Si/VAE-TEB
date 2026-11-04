r"""Linear forecast baselines against the checkpoint, on a recording-level split (session 1).

Baselines (all direct multi-step, fitted by ridge least squares on the "fit" half of recordings):
  * ``own_ar``    per channel: the channel's own last P values -> its H future values;
  * ``trend``     y_t + (tau + 1)(y_t - y_{t-1}), no fit;
  * ``ridge_y``   all kept target channels' last P values -> every (tau, c) cell;
  * ``ridge_yu``  ``ridge_y`` plus the kept source channels' last P values (source zeroed where invalid).
The model's full and base means are scored on the same "eval" half, the same anchors, the same cells.
Only anchors whose whole horizon is valid are used, so every method sees the same cells.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from teb_vae.neural_surgery_trf_cfs import common  # noqa: E402
from teb_vae.lag_attn.config import load_config  # noqa: E402
from teb_vae.lag_attn_cfs.eval.config_schema import force_single_process_loader, merge_eval_overrides  # noqa: E402
from teb_vae.lag_attn_cfs.eval.probe import load_task, read_checkpoint  # noqa: E402
from teb_vae.lag_attn_rws.nets.raw_masks import forecast_mask  # noqa: E402
from teb_vae.lag_attn_transformer_cfs.eval.binding import TRF_CFS_BINDING  # noqa: E402
from train.data_module import GraphDataModule  # noqa: E402

RUN_ARGS = {
    "checkpoint": str(common.CHECKPOINT),
    "config": str(common.CONFIG),
    "overrides": None,            # None: write the default local override file
    "out_dir": str(common.RESULTS / "session1"),
}
P = 6          # history lags per channel
STRIDE = 10    # anchor subsampling, to keep the design matrices small
RIDGE = 1e-3


def is_fit(guid):
    """Assign a recording to the fit half by the parity of its MD5 digest.

    This is the session-1 split. It differs from the CRC32 split of
    :mod:`teb_vae.neural_surgery_trf_cfs.common`.

    Args:
        guid: The recording GUID.

    Returns:
        ``True`` for the fit half.
    """
    g = guid if isinstance(guid, str) else bytes(guid).decode(errors="ignore")
    return int(hashlib.md5(g.encode()).hexdigest(), 16) % 2 == 0


def main(a):
    """Fit the baselines on the fit half; score them and the model on the other half.

    Args:
        a: The run arguments; see :data:`RUN_ARGS`.
    """
    S = Path(a["out_dir"])
    S.mkdir(parents=True, exist_ok=True)
    a = dict(a, overrides=a["overrides"] or str(common.write_overrides()))
    dev = torch.device("cuda:0")
    config = merge_eval_overrides(load_config(a["config"]), a["overrides"])
    force_single_process_loader(config)
    blob = read_checkpoint(a["checkpoint"])
    task = load_task(a["checkpoint"], dev, blob=blob, binding=TRF_CFS_BINDING)
    model = task.orig_model
    lk = model.forecast_likelihood_kwargs()
    cm = lk["cell_mask"].bool()  # (H, C)
    H, C = cm.shape
    keep_u = model.source_gate.keep_index if model.source_gate is not None else None

    store = {"fit": {"X": [], "U": [], "Y": []}, "eval": {"X": [], "U": [], "Y": [], "mf": [], "mb": []}}
    for bi, batch in enumerate(GraphDataModule(config).test_dataloader()):
        batch = task.transfer_batch_to_device(batch, dev, 0)
        guids = list(batch.guid)
        with torch.no_grad():
            inputs = task._build_forward_inputs(batch)
            tf, weight = task._build_raw_target(batch)
            weight = task._target_quality_weight(batch, weight)
            out = model(*inputs)
            w = model.scored_weight(weight)
            anchors, av = out["anchor_index"], out["anchor_valid"]
            tgt = model._build_forecast_target(tf, anchors)  # (B, A, H, C)
            smask, _ = forecast_mask(w, model.geometry, coverage_floor=model.coverage_floor,
                                     anchors=anchors, anchor_valid=av)
            full = (smask > 0).all(-1) & av  # whole horizon valid
            yk = torch.index_select(tf, -1, model.target_gate.keep_index)  # (B, T, C)
            u = inputs[2]
            sv = inputs[5]
            uk = u if keep_u is None else torch.index_select(u, -1, keep_u)
            svk = sv if keep_u is None else torch.index_select(sv, -1, keep_u)
            uk = uk * svk.to(uk.dtype)
            lag_idx = anchors[:, :, None] - torch.arange(P, device=dev)[None, None, :]  # (B, A, P)
            Xl = torch.stack([yk[b][lag_idx[b]] for b in range(yk.shape[0])])  # (B, A, P, C)
            Ul = torch.stack([uk[b][lag_idx[b]] for b in range(uk.shape[0])])
            sel_a = torch.zeros_like(full)
            sel_a[:, ::STRIDE] = True
            for b in range(yk.shape[0]):
                m = full[b] & sel_a[b]
                if not m.any():
                    continue
                part = "fit" if is_fit(guids[b]) else "eval"
                store[part]["X"].append(Xl[b][m].cpu())
                store[part]["U"].append(Ul[b][m].cpu())
                store[part]["Y"].append(tgt[b][m].cpu())
                if part == "eval":
                    store[part]["mf"].append(out["mu_full"][b][m].cpu())
                    store[part]["mb"].append(out["mu_base"][b][m].cpu())
        if bi % 20 == 0:
            print("batch", bi, flush=True)

    cat = {p: {k: torch.cat(v).double() for k, v in d.items()} for p, d in store.items()}
    fit, ev = cat["fit"], cat["eval"]
    cmn = cm.cpu().numpy()
    res = {"n_fit": int(fit["Y"].shape[0]), "n_eval": int(ev["Y"].shape[0])}

    def mse_cells(pred, Y):
        """Per-cell MSE over anchors, shape $(H, C)$."""
        return ((pred - Y) ** 2).mean(0).numpy()  # (H, C)

    preds = {"model_full": ev["mf"], "model_base": ev["mb"],
             "persistence": ev["X"][:, 0, None, :].expand_as(ev["Y"]),
             "climatology_zero": torch.zeros_like(ev["Y"]),
             "trend": ev["X"][:, 0, None, :] + (torch.arange(H, dtype=torch.float64)[None, :, None] + 1)
             * (ev["X"][:, 0, None, :] - ev["X"][:, 1, None, :])}

    # own-channel AR, direct multi-step
    own = torch.zeros_like(ev["Y"])
    for c in range(C):
        Xf = torch.cat([fit["X"][:, :, c], torch.ones(fit["X"].shape[0], 1, dtype=torch.float64)], 1)
        Xe = torch.cat([ev["X"][:, :, c], torch.ones(ev["X"].shape[0], 1, dtype=torch.float64)], 1)
        A_ = Xf.T @ Xf + RIDGE * torch.eye(Xf.shape[1], dtype=torch.float64)
        W = torch.linalg.solve(A_, Xf.T @ fit["Y"][:, :, c])
        own[:, :, c] = Xe @ W
    preds["own_ar"] = own

    def ridge(feat_f, feat_e, lam):
        """Ridge fit on fit-half features and prediction on eval-half features."""
        Xf = torch.cat([feat_f, torch.ones(feat_f.shape[0], 1, dtype=torch.float64)], 1)
        Xe = torch.cat([feat_e, torch.ones(feat_e.shape[0], 1, dtype=torch.float64)], 1)
        A_ = Xf.T @ Xf + lam * Xf.shape[0] * torch.eye(Xf.shape[1], dtype=torch.float64)
        W = torch.linalg.solve(A_, Xf.T @ fit["Y"].reshape(fit["Y"].shape[0], -1))
        return (Xe @ W).reshape(ev["Y"].shape)

    fy, ey = fit["X"].reshape(fit["X"].shape[0], -1), ev["X"].reshape(ev["X"].shape[0], -1)
    fu, eu = fit["U"].reshape(fit["U"].shape[0], -1), ev["U"].reshape(ev["U"].shape[0], -1)
    preds["ridge_y"] = ridge(fy, ey, 1e-4)
    preds["ridge_yu"] = ridge(torch.cat([fy, fu], 1), torch.cat([ey, eu], 1), 1e-4)

    Y = ev["Y"]
    table = {}
    st = np.zeros(C, bool); st[:32] = True
    for name, pr in preds.items():
        m = mse_cells(pr, Y)
        table[name] = {
            "mse_all": float((m * cmn).sum() / cmn.sum()),
            "mse_st": float((m[:, st] * cmn[:, st]).sum() / cmn[:, st].sum()),
            "mse_ph": float((m[:, ~st] * cmn[:, ~st]).sum() / cmn[:, ~st].sum()),
            "mse_tau0_st": float(m[0, st].mean()), "mse_tau0_ph": float(m[0, ~st].mean()),
            "mse_tau29_st": float(m[29, st].mean()),
        }
        np.save(S / f"baseline_mse_{name}.npy", m)
    res["table"] = table
    with open(S / "linear_baselines.json", "w") as f:
        json.dump(res, f, indent=1)
    for k, v in table.items():
        print(f"{k:18s} " + " ".join(f"{kk}={vv:.4f}" for kk, vv in v.items()))
    print(res["n_fit"], res["n_eval"])


if __name__ == "__main__":
    main(RUN_ARGS)
