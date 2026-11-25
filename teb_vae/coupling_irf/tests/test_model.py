"""Three checks of the coupling model on synthetic signals: it reads a planted delay, it stays shut for a
UA that does not belong to the FHR, and it is causal."""
import torch

from teb_vae.coupling_irf.model import CouplingIRF, irf_delay, plant


def synthetic(B=8, T=600, seed=0):
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(B, T, generator=g)
    for t in range(1, T):                                   # slow FHR wander, a few bpm
        noise[:, t] = 0.9 * noise[:, t - 1] + noise[:, t]
    ua = torch.full((B, T), 10.0)
    steps = torch.arange(T)
    for b in range(B):                                      # a contraction every 30-50 steps (2-3 min)
        peak = int(torch.randint(5, 30, (1,), generator=g))
        while peak < T:
            ua[b] += 50 * torch.exp(-0.5 * ((steps - peak) / 5.0) ** 2)
            peak += int(torch.randint(30, 50, (1,), generator=g))
    return dict(fhr=140 + noise, fhr_sd=torch.ones(B, T), fhr_ok=torch.ones(B, T), ua=ua, ua_ok=torch.ones(B, T))


def test_planted_delay_is_read_and_wrong_ua_stays_shut():
    torch.manual_seed(0)
    model = CouplingIRF().eval()                            # the UA path needs no training to do this
    b = synthetic()
    u = model.ua_features(b["ua"], b["ua_ok"])["u"]
    for true_delay in (2.0, 5.0, 9.0, 15.0, 23.0):          # 8, 20, 36, 60 and 92 s
        planted = dict(b, fhr=plant(b["fhr"], u, torch.full((8,), true_delay), depth=20.0))
        with torch.no_grad():
            st, own = model(planted)
            wrong = model.ua_side(st, planted["ua"].roll(1, 0), planted["ua_ok"])
        delay, depth = irf_delay(own["h"][:, -1])           # IRF at the last anchor
        assert (delay - true_delay).abs().median() < 1.0, (true_delay, delay)   # within one step (4 s)
        assert (depth < 0).all()
        assert own["a"][:, -1].mean() > 0.9 > 2 * wrong["a"][:, -1].mean(), (own["a"][:, -1], wrong["a"][:, -1])   # k is untrained here


def test_nothing_at_an_anchor_depends_on_the_future():
    torch.manual_seed(0)
    model = CouplingIRF().eval()
    with torch.no_grad():
        model.head_mu.weight.normal_(std=0.1)
    b = synthetic(seed=1)
    u = model.ua_features(b["ua"], b["ua_ok"])["u"]
    b["fhr"] = plant(b["fhr"], u, torch.full((8,), 8.0))    # coupled, so the UA path is switched on
    cut = 400                                               # change everything after step 400
    later = {k: v.clone() for k, v in b.items()}
    later["fhr"][:, cut + 1:] += 30.0
    later["ua"][:, cut + 1:] += 40.0
    with torch.no_grad():
        st0, o0 = model(b)
        st1, o1 = model(later)
        ua_only = model.ua_side(st0, later["ua"], later["ua_ok"])
        _, o2 = model(dict(b, fhr=later["fhr"]))
    past = st0["anchors"] <= cut
    assert o0["a"][:, past].max() > 0.3
    for key in ("mu0", "ls0"):
        assert torch.allclose(st0[key][:, past], st1[key][:, past], atol=1e-5), key
    for key in ("d_fore", "beta", "a", "h"):
        assert torch.allclose(o0[key][:, past], o1[key][:, past], atol=1e-5), key
    # the "expl" mode reads UA up to the target, so future UA must change it, but future FHR must not
    assert not torch.allclose(o0["d_expl"][:, past], ua_only["d_expl"][:, past], atol=1e-5)
    assert torch.allclose(o0["d_expl"][:, past], o2["d_expl"][:, past], atol=1e-5)
