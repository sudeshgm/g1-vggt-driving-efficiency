"""Co-Me (arXiv 2511.14751, commit 5c7cfbc, CC BY-NC-SA 4.0) — pure-PyTorch port of the OFFICIAL direct mode
(third_party/CoMe/ref/direct.py + co_me_2d.py, config co_me_direct_vggt_3x3.yaml) onto stock VGGT, for sm75 (T4)/CPU.
  * confidence predictor = official GlobalAttentionConfidence + official distilled ckpt (vggt/20260324_172742/step_02000),
    run on encoder block-14 output; Ranking mode ("pairwise-rank"): merge 3x3 blocks whose mean conf < quantile(threshold)
  * merge = official naive_func_token_merge_2d (verbatim, block mean); split = official index_select fallback
  * encoder blocks 15..23, all frame and all global blocks: merge before attention (pos merged too) and before MLP
  * attention = per-segment SDPA with additive per-key bias log(weight) (== their flash_attn varlen w/ perkey bias;
    weight = 9 for merged tokens); global layers coalesce all frames of a batch item into one segment.
Deviations: fused CUDA kernels (co_me_cuext + flash-attn-perkey-bias, sm80+) replaced by the math above (same outputs,
not their speed); fp16 instead of bf16 on T4; token grid must be divisible by 3 -> input resized 294x518 -> 294x504."""
import json, os, types
from dataclasses import dataclass
import torch, torch.nn.functional as F
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "CoMe")
CKPT = os.path.join(ROOT, "ckpt", "vggt", "20260324_172742", "step_02000")

@dataclass
class JaggedTokens:
    tokens: torch.Tensor; weight: torch.Tensor; offset: torch.Tensor

def naive_func_token_merge_2d(tokens, start_index, group_h, group_w, H, W, merge_mask):  # verbatim from co_me_2d.py
    device = tokens.device; B, N, C = tokens.shape; M_h, M_w = H // group_h, W // group_w; M = M_h * M_w; G = group_h * group_w
    k_b = merge_mask.sum(dim=1, dtype=torch.int64); L_b = start_index + (M - k_b) * G + k_b
    offsets = torch.empty((B + 1,), device=device, dtype=torch.int64); offsets[0] = 0; offsets[1:] = torch.cumsum(L_b, dim=0)
    P = int(offsets[-1].item())
    out = torch.empty((P, C), device=device, dtype=tokens.dtype); rev = torch.empty((B, N), device=device, dtype=torch.int64)
    weight = torch.empty((P,), device=device, dtype=torch.float); base = offsets[:-1]
    if start_index > 0:
        sp = base[:, None] + torch.arange(start_index, device=device, dtype=torch.int64)[None, :]
        out.index_copy_(0, sp.reshape(-1), tokens[:, :start_index, :].reshape(-1, C)); weight[sp.reshape(-1)] = 1.0; rev[:, :start_index] = sp
    img = tokens[:, start_index:, :].reshape(B, H, W, C)
    img_blocks = img.reshape(B, M_h, group_h, M_w, group_w, C).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, M, G, C)
    merged_means = img_blocks.mean(dim=2)
    group_len = torch.where(merge_mask, 1, G); group_prefix = torch.cumsum(group_len, dim=1) - group_len
    group_base = base[:, None] + start_index + group_prefix
    unmerged_pos = group_base[..., None] + torch.arange(G, device=device, dtype=torch.int64)
    merged_pos = group_base[merge_mask]; out.index_copy_(0, merged_pos, merged_means[merge_mask]); weight[merged_pos] = float(G)
    um = ~merge_mask; upf = unmerged_pos[um].reshape(-1); out.index_copy_(0, upf, img_blocks[um].reshape(-1, C)); weight[upf] = 1.0
    block_rev = torch.where(merge_mask[..., None], group_base[..., None], unmerged_pos)
    hw = torch.arange(H * W, device=device, dtype=torch.int64).reshape(M_h, group_h, M_w, group_w).permute(0, 2, 1, 3).reshape(M * G)
    rev_img = torch.empty((B, H * W), device=device, dtype=torch.int64)
    rev_img.scatter_(1, hw.unsqueeze(0).expand(B, -1), block_rev.reshape(B, M * G)); rev[:, start_index:] = rev_img
    return JaggedTokens(tokens=out.unsqueeze(0), weight=weight, offset=offsets), rev.flatten()

class GlobalAttentionConfidence(torch.nn.Module):  # official confidence_predictor.py (num_register=0, expp1)
    def __init__(self, hidden_dims, activation, num_register=0):
        super().__init__(); from vggt.layers.block import Block
        self.proj1 = torch.nn.Linear(hidden_dims[0], hidden_dims[1]); self.block = Block(dim=hidden_dims[1], num_heads=1, mlp_ratio=2)
        self.proj2 = torch.nn.Sequential(torch.nn.Conv2d(hidden_dims[1], hidden_dims[1], 3, 1, 1, padding_mode="reflect"), torch.nn.GELU(),
                                         torch.nn.Conv2d(hidden_dims[1], hidden_dims[2], 3, 1, 1, padding_mode="reflect"))
        assert num_register == 0; self.register_token = None; self.activation = activation
    def forward(self, tokens, BPHWC):
        B, P, H, W, C = BPHWC; t = self.proj1(tokens.reshape(B, P * H * W, C)); t = self.block(t)
        t = self.proj2(t.reshape(B * P, H, W, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).flatten(1, 2)
        return torch.exp(t) + 1.0 if self.activation == "expp1" else t

class CoMeMerger:
    def __init__(self, start_idx, threshold, device, dtype, ckpt=CKPT, group=3):
        cfg = json.load(open(os.path.join(ckpt, "config.json"))); assert cfg["loss_function"] == "pairwise-rank"
        self.p = GlobalAttentionConfidence(cfg["predictor_dims"], cfg["predictor_active"], cfg.get("predictor_num_register", 0))
        self.p.load_state_dict(torch.load(os.path.join(ckpt, "checkpoint.pth"), weights_only=True, map_location="cpu"))
        self.p = self.p.to(device=device, dtype=dtype).eval(); self.dtype = dtype
        self.start_idx, self.g, self.threshold = start_idx, group, threshold
    def build_ctx(self, feature):  # Ranking mode of official _CoMe_2D_Base.build_ctx
        B, S, C, H, W = feature.shape; g = self.g; assert H % g == 0 and W % g == 0, (H, W); Mh, Mw = H // g, W // g
        conf = self.p(feature.permute(0, 1, 3, 4, 2).to(self.dtype), BPHWC=(B, S, H, W, C)).squeeze(-1)
        blocks = conf.reshape(B * S, Mh, g, Mw, g).permute(0, 1, 3, 2, 4).reshape(B * S, Mh * Mw, g * g)
        bm = blocks.mean(-1).reshape(B, S * Mh * Mw); pv = torch.quantile(bm.float(), self.threshold, dim=-1, keepdim=True)
        return ((bm < pv).reshape(B * S, Mh * Mw), (B, S, C, H, W))
    def merge(self, ctx, tokens):
        mask, (B, S, C, H, W) = ctx
        m, rev = naive_func_token_merge_2d(tokens.flatten(0, 1), self.start_idx, self.g, self.g, H, W, mask); return m, (rev, tokens.shape)
    def split(self, stub, jt):
        rev, (B, S, N, C) = stub; return jt.tokens.squeeze(0).index_select(0, rev).reshape(B, S, N, C)

def jagged_attention_w_bias(attn, x: JaggedTokens, pos):
    """== official JaggedAttention_w_Bias (flash varlen, per-key bias log w) via per-segment SDPA."""
    t = x.tokens.squeeze(0); P, C = t.shape; qkv = attn.qkv(t).reshape(P, 3, attn.num_heads, attn.head_dim)
    q, k, v = qkv.unbind(1); q, k = attn.q_norm(q), attn.k_norm(k)
    if attn.rope is not None and pos is not None:
        q = attn.rope(q.unsqueeze(0).transpose(1, 2), pos).transpose(1, 2).squeeze(0)
        k = attn.rope(k.unsqueeze(0).transpose(1, 2), pos).transpose(1, 2).squeeze(0)
    bias = x.weight.float().log().to(q.dtype); out = torch.empty_like(q); off = x.offset.tolist()
    for a, b in zip(off[:-1], off[1:]):
        qs, ks, vs = (z[a:b].transpose(0, 1).unsqueeze(0) for z in (q, k, v))
        m = bias[a:b].view(1, 1, 1, -1); L = b - a; CH = 4096  # query chunks keep the (broadcast) bias mask small on T4
        o = torch.cat([F.scaled_dot_product_attention(qs[:, :, i:i + CH], ks, vs, attn_mask=m.expand(1, 1, min(CH, L - i), L), scale=attn.scale)
                       for i in range(0, L, CH)], 2)
        out[a:b] = o.squeeze(0).transpose(0, 1)
    return JaggedTokens(attn.proj_drop(attn.proj(out.reshape(P, C))).unsqueeze(0), x.weight, x.offset)

def apply_come(model, threshold=0.5, encoder_layer=14, dtype=torch.float32, device="cpu"):
    agg = model.aggregator; blocks = agg.patch_embed.blocks
    merger = CoMeMerger(agg.patch_start_idx, threshold, device, dtype); st = {}
    def pre(module, args):
        B, S, C, H, W = args[0].shape; st["shape"] = (B, S, module.patch_embed.num_features, H // 14, W // 14)
    agg.register_forward_pre_hook(pre)
    def hook(module, args, output):
        st["ctx"] = merger.build_ctx(output[:, agg.patch_start_idx:].permute(0, 2, 1).reshape(*st["shape"]))
        st["merged_frac"] = float(st["ctx"][0].float().mean())
    blocks[encoder_layer].register_forward_hook(hook)
    def attn_fwd(is_global):
        def f(self, x, pos=None, **_):
            B, S, C, _, _ = st["shape"]; ish, idt = x.shape, x.dtype
            mx, rev = merger.merge(st["ctx"], x.reshape(B, S, -1, C))
            if pos is not None: pm, _ = merger.merge(st["ctx"], pos.reshape(B, S, -1, 2).to(x.dtype)); pos = pm.tokens.long()
            if is_global: mx.offset = mx.offset[torch.arange(0, B * S + 1, S, device=mx.offset.device)]
            return merger.split(rev, jagged_attention_w_bias(self, mx, pos)).view(*ish).to(idt)
        return f
    def mlp_fwd(orig):
        def f(self, x):
            B, S, C, _, _ = st["shape"]; ish = x.shape; mx, rev = merger.merge(st["ctx"], x.reshape(B, S, -1, C))
            mx.tokens = orig(mx.tokens); return merger.split(rev, mx).view(*ish)
        return f
    for blk, g in [(b, False) for b in list(blocks)[encoder_layer + 1:]] + [(b, False) for b in agg.frame_blocks] + [(b, True) for b in agg.global_blocks]:
        blk.attn.forward = types.MethodType(attn_fwd(g), blk.attn); blk.mlp.forward = types.MethodType(mlp_fwd(blk.mlp.forward), blk.mlp)
    model._come_state = st; return model
