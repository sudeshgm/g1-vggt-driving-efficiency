"""Analytic FLOP model for VGGT-1B (facebook/VGGT-1B). 1 MAC = 2 FLOPs.
Architecture (VGGT paper / repo): DINOv2 ViT-L/14 patch embed+24 blocks (dim 1024, 16 heads, MLP 4x, +1 cls +4 reg);
aggregator: 24 frame-attention + 24 global-attention blocks, dim 1024, MLP 4x, per-frame tokens = patches + 1 camera + 4 register.
Heads: camera head (4-block trunk on camera tokens, x4 iterations) and two DPT heads (depth, point) -- DPT counted
approximately (labelled). Encoder & aggregator counts are exact matmul counts; softmax/LN/GELU ignored (<1%).
"""
from dataclasses import dataclass

D, MLP, L_ENC, L_AGG, PATCH = 1024, 4096, 24, 24, 14
N_SPECIAL_ENC, N_SPECIAL_AGG = 5, 5

def block_flops(n_tok, d=D, mlp=MLP, attn_tok=None):
    """One transformer block over n_tok query tokens attending attn_tok keys (default all)."""
    a = n_tok if attn_tok is None else attn_tok
    qkv = 2 * n_tok * d * 3 * d
    attn = 2 * n_tok * a * d * 2          # QK^T and AV
    proj = 2 * n_tok * d * d
    mlpf = 2 * n_tok * d * mlp * 2
    return qkv + attn + proj + mlpf

def grid(h, w):
    return (h // PATCH) * (w // PATCH)

def input_hw(scale=1.0, src_hw=(900, 1600), long_side=518):
    """VGGT 'crop' preprocessing: width=518, height keeps aspect, rounded to multiple of 14."""
    W = int(round(long_side * scale / PATCH)) * PATCH
    H = int(round(W * src_hw[0] / src_hw[1] / PATCH)) * PATCH
    return H, W

def dpt_flops(n_frames, P, d=D):
    # Approximate: 4 projection layers + fusion convs at 256 ch over upsampled maps (DPT-L style).
    # ~= 4*2*P*d*256 + fusion ~ 2*(P*4+P*16+P*64)*256*256*9*2 per head. Labelled ESTIMATE.
    proj = 4 * 2 * P * d * 256
    fus = 2 * (P * (1 + 4 + 16 + 64)) * 256 * 256 * 9 * 2 / 4
    return n_frames * (proj + fus)

@dataclass
class Cfg:
    n_frames: int = 32
    scale: float = 1.0
    keep_global: float = 1.0      # token keep ratio in global blocks (merging methods)
    kv_keep_global: float = 1.0   # key/value subsampling ratio in global attn (AVGGT, sparse)
    n_global_as_frame: int = 0    # AVGGT: early global layers converted to frame attention
    include_heads: bool = True

def vggt_flops(c: Cfg, src_hw=(900, 1600)):
    H, W = input_hw(c.scale, src_hw)
    P = grid(H, W); N = c.n_frames
    enc = N * (L_ENC * block_flops(P + N_SPECIAL_ENC) + 2 * P * (3 * PATCH * PATCH) * D)
    t = P + N_SPECIAL_AGG
    frame = N * L_AGG * block_flops(t)
    g_tok = int(round(N * t * c.keep_global))
    g_kv = int(round(g_tok * c.kv_keep_global))
    n_g = L_AGG - c.n_global_as_frame
    glob = n_g * block_flops(g_tok, attn_tok=g_kv) + c.n_global_as_frame * N * block_flops(t)
    heads = 0
    if c.include_heads:
        heads = 4 * 4 * N * block_flops(1, d=2 * D, mlp=8 * D) + 2 * dpt_flops(N, P)
    return dict(H=H, W=W, patches=P, encoder=enc, frame=frame, global_=glob, heads_est=heads,
                total=enc + frame + glob + heads)

def ratio(c: Cfg, ref: Cfg = Cfg()):
    return vggt_flops(c)["total"] / vggt_flops(ref)["total"]

def solve_keep(budget, n_frames=32, field="keep_global", **kw):
    """Bisection on a keep ratio to hit budget fraction of dense N=32 full-res. Returns (x, achieved) or (None, min)."""
    lo, hi = 0.0, 1.0
    f = lambda x: ratio(Cfg(n_frames=n_frames, **{field: x}, **kw))
    if f(0.0) > budget: return None, f(0.0)
    for _ in range(60):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if f(m) < budget else (lo, m)
    return lo, f(lo)
