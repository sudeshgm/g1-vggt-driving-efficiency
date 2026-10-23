"""Method-specific FLOP models for the accelerators, built on g1/flops.py conventions (1 MAC = 2 FLOPs; heads = estimate).
Key point vs the earlier plan: FastVGGT, GoToHunt and AVGGT shrink ONLY the global-attention matmuls (QK^T, AV);
qkv/proj/MLP still run on every token. Co-Me merges tokens before attention AND MLP in encoder blocks >14 and in all
frame/global blocks. Fractions are relative to dense VGGT N=32 at 294x518 (g1.flops.vggt_flops(Cfg()))."""
from functools import lru_cache
from g1.flops import D, MLP, L_ENC, L_AGG, PATCH, N_SPECIAL_AGG, input_hw, grid, dpt_flops, vggt_flops, Cfg

def lin(n, d=D, mlp=MLP): return 2 * n * d * 3 * d + 2 * n * d * d + 2 * n * d * mlp * 2
def att(nq, nk, d=D): return 4 * nq * nk * d
REF = vggt_flops(Cfg())["total"]

def _parts(N, H, W, enc_tok=None, frame_tok=None, glob=None, extra=0.0):
    P = (H // PATCH) * (W // PATCH); t = P + N_SPECIAL_AGG
    enc = N * (L_ENC * (lin(t) + att(t, t)) + 2 * P * 3 * PATCH * PATCH * D) if enc_tok is None else enc_tok
    frame = N * L_AGG * (lin(t) + att(t, t)) if frame_tok is None else frame_tok
    g = L_AGG * (lin(N * t) + att(N * t, N * t)) if glob is None else glob
    heads = 4 * 4 * N * (lin(1, 2 * D, 8 * D) + att(1, 1, 2 * D)) + 2 * dpt_flops(N, P)
    tot = enc + frame + g + heads + extra
    return dict(total=tot, frac=tot / REF, encoder=enc, frame=frame, global_=g, heads_est=heads, method_overhead=extra)

def dense(N=32, scale=1.0): return _parts(N, *input_hw(scale))

# ---------------- FastVGGT (official code: merging/merge.py, vggt/layers/attention.py) ----------------
@lru_cache(None)
def fastvggt_kept(N, h, w, merge_ratio):
    """Exact token count after token_merge_bipartite2d (2x2 dst grid, first frame + specials = dst, 10% protected),
    obtained by running the official function on a dummy metric (count only)."""
    import torch, sys, os
    sys.path.insert(0, os.environ.get("FASTVGGT_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "FastVGGT")))
    from merging.merge import token_merge_bipartite2d
    n = N * (h * w + 5); x = torch.randn(1, n, 8)
    g = torch.Generator().manual_seed(33)
    m, _ = token_merge_bipartite2d(x, w, h, 2, 2, int(n * merge_ratio), False, g, enable_protection=True)
    out = m(x, mode="mean"); out = out[0] if isinstance(out, tuple) else out
    return out.shape[1], n

def fastvggt(N=32, scale=1.0, merge_ratio=0.9, merging=0):
    H, W = input_hw(scale); h, w = H // PATCH, W // PATCH; Nt = N * (h * w + 5)
    M, _ = fastvggt_kept(N, h, w, merge_ratio)
    n_dst = (h * w + 5) + (N - 1) * (5 + (h // 2) * (w // 2)); n_src = Nt - n_dst
    g = sum(lin(Nt) + (att(M, M) if l >= merging else att(Nt, Nt)) for l in range(L_AGG))
    sim = sum(2 * n_src * n_dst * D for l in range(L_AGG) if l >= merging)  # bipartite similarity (metric = x, C=1024)
    r = _parts(N, H, W, glob=g, extra=sim); r.update(kept_tokens=M, total_tokens=Nt); return r

# ---------------- GoToHunt (official attention.py, standalone token_downsample path, frame selection OFF) ----------------
def gotohunt_ctx(h, w, t):
    """Kept K/V per frame with patch_start_idx=0 (official default ds_keep_register=False): strided grid only."""
    return len(range(0, h, t)) * len(range(0, w, t))

def gotohunt(N=32, scale=1.0, t=3, ds_layers=range(2, 9), gaf_layers=(0, 1)):
    H, W = input_hw(scale); h, w = H // PATCH, W // PATCH; tf = h * w + 5; Nt = N * tf
    k = N * gotohunt_ctx(h, w, t); g = 0
    for l in range(L_AGG):
        if l in gaf_layers: g += N * (lin(tf) + att(tf, tf))
        elif l in ds_layers: g += lin(Nt) + att(Nt, k)
        else: g += lin(Nt) + att(Nt, Nt)
    return _parts(N, H, W, glob=g)

# ---------------- Co-Me (official direct_accelerate: merge before attn+MLP, encoder blocks > 14, all aggregator blocks) ----
def come_hw(scale=1.0, g=3):
    """Co-Me needs token grid divisible by 3: width 37->36 tokens (504 px) at full res (we resize, not crop)."""
    H, W = input_hw(scale); h, w = (H // PATCH) // g * g, (W // PATCH) // g * g; return h * PATCH, w * PATCH

def come(N=32, scale=1.0, q=0.5, enc_layer=14, g=3):
    H, W = come_hw(scale, g); h, w = H // PATCH, W // PATCH; P = h * w; M = P // (g * g)
    k = int(round(q * M)); tm = N_SPECIAL_AGG + (M - k) * g * g + k; t = P + N_SPECIAL_AGG
    enc = N * ((enc_layer + 1) * (lin(t) + att(t, t)) + (L_ENC - enc_layer - 1) * (lin(tm) + att(tm, tm)) + 2 * P * 3 * PATCH * PATCH * D)
    frame = N * L_AGG * (lin(tm) + att(tm, tm)); glob = L_AGG * (lin(N * tm) + att(N * tm, N * tm))
    d2 = 128; NP = N * P   # confidence predictor: proj1, 1 global block (dim128, 1 head, mlp x2), two 3x3 convs
    pred = 2 * NP * D * d2 + lin(NP, d2, 2 * d2) + att(NP, NP, d2) + 2 * NP * d2 * d2 * 9 + 2 * NP * d2 * 9
    r = _parts(N, H, W, enc_tok=enc, frame_tok=frame, glob=glob, extra=pred); r.update(tokens_per_frame=tm, hw=(H, W)); return r

# ---------------- AVGGT (our reimplementation; g1/avggt.py) ----------------
def avggt(N=32, scale=1.0, sigma=4, t_early=9):
    from g1.avggt import kept_key_index
    H, W = input_hw(scale); h, w = H // PATCH, W // PATCH; tf = h * w + 5; Nt = N * tf
    nk = len(kept_key_index(N, tf, N_SPECIAL_AGG, (h, w), sigma)) + 1  # + 1 mean K/V (diagonal term ~ negligible)
    g = sum(N * (lin(tf) + att(tf, tf)) if l < t_early else lin(Nt) + att(Nt, nk) for l in range(L_AGG))
    return _parts(N, H, W, glob=g)
