import numpy as np
from g1.flops import *

def test_input_hw_nuscenes():
    assert input_hw(1.0) == (294, 518) and grid(294, 518) == 21 * 37

def test_block_flops_hand_count():
    n, d = 10, 4
    assert block_flops(n, d=d, mlp=16) == 2*n*d*3*d + 4*n*n*d + 2*n*d*d + 4*n*d*16

def test_global_attention_quadratic_in_frames():
    a = vggt_flops(Cfg(n_frames=32, include_heads=False)); b = vggt_flops(Cfg(n_frames=16, include_heads=False))
    t = a["patches"] + 5
    attn32 = 24 * 4 * (32 * t) ** 2 * D; attn16 = 24 * 4 * (16 * t) ** 2 * D
    assert np.isclose(a["global_"] - attn32, 2 * (b["global_"] - attn16))  # linear part halves
    assert np.isclose(attn32 / attn16, 4)

def test_frame_drop_linear_parts_halve():
    a = vggt_flops(Cfg(32)); b = vggt_flops(Cfg(16))
    assert np.isclose(a["encoder"], 2 * b["encoder"]) and np.isclose(a["frame"], 2 * b["frame"])
    assert ratio(Cfg(16)) < 0.5

def test_keep_ratio_one_is_dense_and_monotone():
    assert ratio(Cfg(keep_global=1.0)) == 1.0
    assert ratio(Cfg(keep_global=0.5)) < ratio(Cfg(keep_global=0.8)) < 1

def test_avgggt_all_frame_equals_no_global_cost():
    c = vggt_flops(Cfg(n_global_as_frame=24)); assert np.isclose(c["global_"], c["frame"])

def test_solver_hits_budget():
    x, r = solve_keep(0.5); assert x is not None and abs(r - 0.5) < 1e-3
