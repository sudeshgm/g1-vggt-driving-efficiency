import torch, math
from g1.avggt import sga_attention, kept_key_index, avggt_kv_keep

def ref(q, k, v, keep):
    T = q.shape[2]; D = q.shape[-1]
    m = torch.zeros(T, dtype=torch.bool); m[keep] = True
    L = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    Ls = L[..., m]; diag = torch.diagonal(L, dim1=-2, dim2=-1).unsqueeze(-1).masked_fill(m.view(1, 1, -1, 1), -float("inf"))
    km = k[:, :, ~m].mean(2, keepdim=True); vm = v[:, :, ~m].mean(2, keepdim=True)
    Lm = q @ km.transpose(-1, -2) / math.sqrt(D)
    w = torch.cat([Ls, diag, Lm], -1).softmax(-1)
    return w[..., :m.sum()] @ v[:, :, m] + w[..., -2:-1] * v + w[..., -1:] * vm

def test_keep_index_counts():
    keep = kept_key_index(S=3, P=5 + 12, n_special=5, grid_hw=(3, 4), sigma=4)
    # frame0 all 17, frames 1-2: 5 special + 2x2 grid of 3x4 -> rows{0,2} cols{0,2} = 4
    assert len(keep) == 17 + 2 * (5 + 4)

def test_sga_matches_explicit_reference():
    torch.manual_seed(0); S, gh, gw = 3, 4, 6; P = 5 + gh * gw
    q, k, v = (torch.randn(1, 2, S * P, 8) for _ in range(3))
    keep = kept_key_index(S, P, 5, (gh, gw), 6)
    assert torch.allclose(sga_attention(q, k, v, keep, chunk=7), ref(q, k, v, keep), atol=1e-5)

def test_sigma1_is_dense():
    torch.manual_seed(1); q, k, v = (torch.randn(1, 2, 40, 8) for _ in range(3))
    out = sga_attention(q, k, v, torch.arange(40))
    assert torch.allclose(out, torch.nn.functional.scaled_dot_product_attention(q, k, v), atol=1e-5)

def test_kv_keep_ratio_nuscenes():
    r = avggt_kv_keep(32, (21, 37), 4)
    assert 0.25 < r < 0.30
