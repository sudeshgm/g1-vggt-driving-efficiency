"""GoToHunt adapter (frame/diverse selection OFF) on stock VGGT. Uses the OFFICIAL GoToHunt Attention.forward
(third_party/gotohunt/.../layers/attention.py, commit 6c92c4f, pure torch) bound onto the stock global-attention modules,
with the kwargs the official aggregator passes in its standalone path (no frame_selector => token_ds_layers=None =>
K/V stride-t downsampling in EVERY global layer that is not global-as-frame; psi=0 as ds_keep_register=False).
global_as_frame layers run the global block weights with the per-frame layout (official _process_global_as_frame_attention)."""
import importlib.util, os, types
_P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "gotohunt", "mapanything",
                  "models", "external", "vggt", "layers", "attention.py")

def _official_attention():
    spec = importlib.util.spec_from_file_location("gotohunt_attention", _P); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m.Attention

def apply_gotohunt(model, t=3, gaf_layers=(0, 1), ds_layers=None, ds_keep_register=False):
    GA = _official_attention(); agg = model.aggregator; orig = agg._process_global_attention; gaf = set(gaf_layers)

    def patched(self, tokens, B, S, P, C, global_idx, pos=None):
        if global_idx in gaf:
            tokens = tokens.view(B * S, P, C); pf = pos.view(B * S, P, 2) if pos is not None else None
            tokens = self.global_blocks[global_idx](tokens, pos=pf); return tokens, global_idx + 1, [tokens.view(B, S, P, C)]
        H, W = self._g1_hw; kw = dict(num_frames=S, tokens_per_frame=P, patch_start_idx=self.patch_start_idx if ds_keep_register else 0,
            token_downsample=t if (ds_layers is None or global_idx in ds_layers) else 1, token_keep_rate=1.0,
            token_selection_method="diverse", tokens_h=H // self.patch_size, tokens_w=W // self.patch_size)
        if S <= 1: kw["token_downsample"] = 1  # official: standalone path only when S > 1
        att = self.global_blocks[global_idx].attn
        att.forward = types.MethodType(lambda s, x, pos=None, **_: GA.forward(s, x, pos=pos, **kw), att)
        return orig(tokens, B, S, P, C, global_idx, pos=pos)

    agg._process_global_attention = types.MethodType(patched, agg); of = agg.forward
    def fwd(self, images, *a, **k): self._g1_hw = images.shape[-2:]; return of(images, *a, **k)
    agg.forward = types.MethodType(fwd, agg); return model
