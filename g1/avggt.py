"""AVGGT — OUR REIMPLEMENTATION (no official code located), following arXiv 2512.02541 Sec. 4:
 (1) Global-to-Frame: global blocks with index < t_early (default 9 => 0..8) run as frame attention
     (skip the (B,S*P,C) rearrangement; same weights, per-frame RoPE positions).
 (2) Subsampled Global Attention on remaining global blocks: all queries kept; K/V = all special tokens
     (camera+registers) of every frame + ALL patch tokens of frame 0 (reference, kept uncompressed for VGGT)
     + grid-subsampled patches of frames >= 1 (first patch of each s_h x s_w window; sigma 2->(1,2), 4->(2,2),
     6->(2,3), 9->(3,3)); plus diagonal self term and ONE mean K/V over all dropped patches, all under a
     shared softmax.
Ambiguities (recorded, our assumption): the mean component is a single unweighted logit (no log-count bias);
the diagonal term is only added for queries whose own key was dropped (otherwise it would be double-counted);
special tokens of all frames are kept (paper: 'keeping all Queries and special tokens').
Attention computed explicitly in query chunks (masking-style) => accuracy is exact to the method, speed is NOT
representative; report FLOPs analytically."""
import types, torch

SIGMA_HW = {1: (1, 1), 2: (1, 2), 4: (2, 2), 6: (2, 3), 9: (3, 3)}

def kept_key_index(S, P, n_special, grid_hw, sigma):
    """Indices (into S*P flattened tokens) of retained keys."""
    sh, sw = SIGMA_HW[sigma]; gh, gw = grid_hw
    r = torch.arange(gh).view(-1, 1); c = torch.arange(gw).view(1, -1)
    sel = ((r % sh == 0) & (c % sw == 0)).flatten().nonzero().flatten() + n_special
    idx = [torch.arange(P)]  # frame 0 full
    for s in range(1, S):
        idx.append(torch.cat([torch.arange(n_special), sel]) + s * P)
    return torch.cat(idx)

def sga_attention(q, k, v, keep, chunk=2048):
    """q,k,v: (B,H,T,D). keep: LongTensor of retained key indices. Returns (B,H,T,D)."""
    B, H, T, D = q.shape; scale = D ** -0.5
    mask_keep = torch.zeros(T, dtype=torch.bool, device=q.device); mask_keep[keep] = True
    dropped = ~mask_keep
    kS, vS = k[:, :, keep], v[:, :, keep]
    has_mean = bool(dropped.any())
    if has_mean:
        km = k[:, :, dropped].mean(2, keepdim=True); vm = v[:, :, dropped].mean(2, keepdim=True)
    out = torch.empty_like(q)
    for a in range(0, T, chunk):
        b = min(T, a + chunk); qc = q[:, :, a:b].float() * scale
        parts = [qc @ kS.float().transpose(-1, -2)]
        diag = (qc * k[:, :, a:b].float()).sum(-1, keepdim=True)
        diag = diag.masked_fill(mask_keep[a:b].view(1, 1, -1, 1), float("-inf"))
        parts.append(diag)
        if has_mean: parts.append(qc @ km.float().transpose(-1, -2))
        w = torch.cat(parts, -1).softmax(-1)
        nS = kS.shape[2]
        o = w[..., :nS] @ vS.float() + w[..., nS:nS + 1] * v[:, :, a:b].float()
        if has_mean: o = o + w[..., nS + 1:nS + 2] * vm.float()
        out[:, :, a:b] = o.to(q.dtype)
    return out

def _sga_forward(self, x, pos=None):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0); q, k = self.q_norm(q), self.k_norm(k)
    if self.rope is not None: q = self.rope(q, pos); k = self.rope(k, pos)
    x = sga_attention(q, k, v, self._g1_keep.to(x.device))
    return self.proj(x.transpose(1, 2).reshape(B, N, C))

def apply_avggt(model, sigma=4, t_early=9):
    agg = model.aggregator
    agg._g1 = dict(sigma=sigma, t_early=t_early)
    orig = agg._process_global_attention

    def patched(self, tokens, B, S, P, C, global_idx, pos=None):
        if global_idx < t_early:   # Global-to-Frame conversion: same block, per-frame layout
            tokens = tokens.view(B * S, P, C); pf = pos.view(B * S, P, 2) if pos is not None else None
            tokens = self.global_blocks[global_idx](tokens, pos=pf)
            return tokens, global_idx + 1, [tokens.view(B, S, P, C)]
        att = self.global_blocks[global_idx].attn
        if sigma > 1 and not getattr(att, "_g1_patched", False):
            att.forward = types.MethodType(_sga_forward, att); att._g1_patched = True
        if sigma > 1:
            H, W = self._g1_hw; ps = self.patch_size if hasattr(self, "patch_size") else 14
            att._g1_keep = kept_key_index(S, P, self.patch_start_idx, (H // ps, W // ps), sigma)
        return orig(tokens, B, S, P, C, global_idx, pos=pos)

    agg._process_global_attention = types.MethodType(patched, agg)
    of = agg.forward
    def fwd(self, images, *a, **kw):
        self._g1_hw = images.shape[-2:]; return of(images, *a, **kw)
    agg.forward = types.MethodType(fwd, agg)
    return model

def avggt_kv_keep(S, grid_hw, sigma, n_special=5):
    P = grid_hw[0] * grid_hw[1] + n_special
    return len(kept_key_index(S, P, n_special, grid_hw, sigma)) / (S * P)
