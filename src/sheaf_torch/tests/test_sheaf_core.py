"""Регрессионные тесты PyTorch-порта Sheaf-ADMM.

Запуск:  PYTHONPATH=. .venv/bin/python -m pytest src/sheaf_torch/tests -q
(или без pytest:  PYTHONPATH=. .venv/bin/python src/sheaf_torch/tests/test_sheaf_core.py)
"""

import torch

from src.losses.sheaf_admm_head import SheafADMMHead
from src.sheaf_torch import SheafADMMModule
from src.sheaf_torch.core import EdgeGeometry, GridGraph


def _module(**kw):
    base = dict(in_channels=32, num_classes=7, d_v=24, d_e=12,
                enc_hidden=64, num_iters=5, loss_window=2)
    base.update(kw)
    return SheafADMMModule(**base)


def test_forward_backward_shapes():
    torch.manual_seed(0)
    B, C, H, W = 3, 32, 6, 6
    for kw in (dict(rm_mode="context", z_mode="prox"),
               dict(rm_mode="context", z_mode="project"),
               dict(rm_mode="fixed", z_mode="prox"),
               dict(rm_mode="context", z_mode="prox", connectivity=4),
               dict(rm_mode="context", z_mode="prox", lora_use_gate=True)):
        m = _module(**kw)
        x = torch.randn(B, C, H, W, requires_grad=True)
        logits_w, agg = m(x)
        assert logits_w.shape == (2, B, H * W, 7)
        assert agg.shape == (B, 7)
        (agg.mean() + logits_w.mean()).backward()
        assert torch.isfinite(agg).all()
        assert x.grad is not None and x.grad.norm() > 0
        assert m.base_maps.grad is not None and m.base_maps.grad.abs().sum() > 0


def test_laplacian_is_psd_and_symmetric():
    """L_F = F^T F: <z, L z> >= 0 и <a, L b> == <L a, b> (самосопряжённость)."""
    torch.manual_seed(1)
    dev = "cpu"
    g = GridGraph(4, 4, connectivity=8, device=dev)
    d_v, d_e = 8, 5
    R = torch.randn(g.num_directions, d_e, d_v)
    geo = EdgeGeometry(g, R[g.dir_uv], R[g.dir_vu])  # fixed maps (shared по направлению)
    a = torch.randn(2, g.N, d_v)
    b = torch.randn(2, g.N, d_v)
    La = geo.laplacian_apply(a)
    Lb = geo.laplacian_apply(b)
    quad = (a * La).sum(dim=(1, 2))
    assert (quad >= -1e-5).all(), quad
    sym = (a * Lb).sum() - (La * b).sum()
    assert sym.abs() < 1e-3, sym.item()


def test_consensus_reduces_disagreement():
    """z-update (prox) должен уменьшать sheaf-рассогласование относительно входа."""
    torch.manual_seed(2)
    m = _module(rm_mode="fixed", z_mode="prox", cg_iters=8, num_iters=6)
    x = torch.randn(2, 32, 6, 6)
    enc, graph, B, N = m._encode(x)
    geo = m._build_geometry(graph, enc, B)
    z0 = enc["h"]
    from src.sheaf_torch.core import z_update_cg
    z1 = z_update_cg(z0 + 0.0, geo, m.rho, "prox", m.gamma, 8, m.tikhonov_eps)
    assert geo.consistency_rms(z1).mean() <= geo.consistency_rms(z0).mean() + 1e-6


def test_head_contract():
    torch.manual_seed(3)
    head = SheafADMMHead(n_class=7, in_channels=32, d_v=24, d_e=12, enc_hidden=64,
                         num_iters=5, loss_window=2)
    x = torch.randn(3, 32, 6, 6, requires_grad=True)
    y = torch.randint(0, 7, (3,))
    out = head(x, y)
    assert "total_loss" in out
    out["total_loss"].backward()
    assert x.grad is not None
    logits = head.predict_logits(x.detach())
    assert logits.shape == (3, 7)


if __name__ == "__main__":
    test_forward_backward_shapes()
    test_laplacian_is_psd_and_symmetric()
    test_consensus_reduces_disagreement()
    test_head_contract()
    print("all sheaf_torch tests passed")
