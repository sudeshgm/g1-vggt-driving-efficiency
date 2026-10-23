import torch
from g1 import flops_accel as FA
from g1.come import naive_func_token_merge_2d, jagged_attention_w_bias

def test_dense_matches_reference():
    assert abs(FA.dense(32)["frac"] - 1.0) < 1e-9

def test_kv_only_methods_keep_linear_floor():
    assert FA.fastvggt(32, merge_ratio=0.99)["frac"] > 0.45 and FA.gotohunt(32, t=8, ds_layers=range(2, 24))["frac"] > 0.45

def test_come_frac_monotone():
    assert FA.come(32, q=0.0)["frac"] > FA.come(32, q=0.5)["frac"] > FA.come(32, q=1.0)["frac"]

def test_come_merge_split_roundtrip_and_counts():
    B, H, W, C, si = 2, 6, 9, 4, 5; x = torch.randn(B, si + H * W, C)
    mask = torch.zeros(B, 6, dtype=torch.bool); mask[0, 1] = True; mask[1, [0, 5]] = True
    jt, rev = naive_func_token_merge_2d(x, si, 3, 3, H, W, mask)
    assert jt.tokens.shape[1] == 2 * (si + H * W) - 3 * 8 and float(jt.weight.max()) == 9.0
    y = jt.tokens.squeeze(0).index_select(0, rev).reshape(B, -1, C)
    um = torch.ones(B, si + H * W, dtype=torch.bool); assert torch.allclose(y[:, :si], x[:, :si])
    jt0, rev0 = naive_func_token_merge_2d(x, si, 3, 3, H, W, torch.zeros_like(mask))
    assert torch.allclose(jt0.tokens.squeeze(0).index_select(0, rev0).reshape(B, -1, C), x)

def test_bias_attention_equals_duplicated_keys():
    """log(w) per-key bias == attending to w identical copies of that key (Co-Me's correction)."""
    from vggt.layers.attention import Attention
    torch.manual_seed(0); a = Attention(16, num_heads=2).eval(); t = torch.randn(1, 5, 16)
    from g1.come import JaggedTokens
    w = torch.tensor([1., 3., 1., 1., 1.]); out = jagged_attention_w_bias(a, JaggedTokens(t, w, torch.tensor([0, 5])), None).tokens
    td = torch.cat([t[:, :1], t[:, 1:2].repeat(1, 3, 1), t[:, 2:]], 1)
    ref = a(td); ref = torch.cat([ref[:, :1], ref[:, 1:2], ref[:, 4:]], 1)
    assert torch.allclose(out, ref, atol=1e-5)
