"""Run stock VGGT-1B (or our AVGGT reimplementation) over manifest windows.
T4/sm75: --dtype fp16 (no bf16), attention via torch SDPA (mem-efficient kernel on sm75; no FlashAttention-2),
--attn xformers as fallback. Weights: facebook/VGGT-1B, NON-COMMERCIAL licence, research use only.
Writes <out>/<window>.npz (idx, extrinsic world->cam 3x4, intrinsic, depth fp16 (S,H,W), depth_conf, input_hw,
secs, peak_mem) and <out>/run_meta.json (gpu name, torch/cuda versions, config, per-window timing).
Example: python scripts/run_vggt.py --stride 2 --out preds/dense_N16"""
import argparse, json, os, sys, time, platform, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="manifests/windows.json"); ap.add_argument("--img_root", default="bundle/images")
ap.add_argument("--stride", type=int, default=1); ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--out", required=True); ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
ap.add_argument("--attn", default="sdpa", choices=["sdpa", "xformers", "math"])
ap.add_argument("--method", default="dense", choices=["dense", "avggt"]); ap.add_argument("--sigma", type=int, default=4)
ap.add_argument("--t_early", type=int, default=9)
ap.add_argument("--max_windows", type=int, default=0); ap.add_argument("--max_frames", type=int, default=0, help="dry-run only")
ap.add_argument("--random_init", action="store_true", help="dry-run only: skip weight download")
ap.add_argument("--warmup", type=int, default=1)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dt = torch.float16 if (a.dtype == "fp16" and dev == "cuda") else torch.float32

if a.attn != "sdpa":
    import vggt.layers.attention as A
    def fwd(self, x, pos=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0); q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None: q = self.rope(q, pos); k = self.rope(k, pos)
        if a.attn == "xformers":
            import xformers.ops as xo
            x = xo.memory_efficient_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).reshape(B, N, C)
        else:
            x = ((q * self.scale) @ k.transpose(-2, -1)).softmax(-1) @ v; x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
    A.Attention.forward = fwd

model = VGGT() if a.random_init else VGGT.from_pretrained("facebook/VGGT-1B")
model = model.to(dev).eval()
if a.method == "avggt":
    from g1.avggt import apply_avggt; apply_avggt(model, sigma=a.sigma, t_early=a.t_early)

meta = dict(config=vars(a), device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor() or "cpu",
            capability=list(torch.cuda.get_device_capability(0)) if dev == "cuda" else None,
            torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version() if dev == "cuda" else None,
            sdpa_flash_enabled=torch.backends.cuda.flash_sdp_enabled() if dev == "cuda" else None,
            dtype=str(dt), weights="random_init(DRY RUN)" if a.random_init else "facebook/VGGT-1B", windows={})
W = json.load(open(a.manifest))["windows"]
if a.max_windows: W = W[:a.max_windows]
def sync():
    if dev == "cuda": torch.cuda.synchronize()
for wi, w in enumerate(W):
    p = f"{a.out}/{w['window_id']}.npz"
    if os.path.exists(p): continue
    n = a.max_frames or w["n"]
    idx = list(range(0, n, a.stride)); files = [os.path.join(a.img_root, w["frames"][i]["file"]) for i in idx]
    imgs = load_and_preprocess_images(files).to(dev)
    if a.scale != 1.0:
        H = max(14, int(round(imgs.shape[2] * a.scale / 14)) * 14); Wd = max(14, int(round(imgs.shape[3] * a.scale / 14)) * 14)
        imgs = F.interpolate(imgs, size=(H, Wd), mode="bilinear", antialias=True, align_corners=False)
    reps = 1 + (a.warmup if wi == 0 else 0)
    for r in range(reps):
        if dev == "cuda": torch.cuda.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast(dev, dtype=dt, enabled=(dt != torch.float32)):
            out = model(imgs)
        sync(); secs = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() if dev == "cuda" else -1
    ext, intr = pose_encoding_to_extri_intri(out["pose_enc"], imgs.shape[-2:])
    np.savez_compressed(p, idx=np.array(idx), extrinsic=ext[0].float().cpu().numpy(), intrinsic=intr[0].float().cpu().numpy(),
        depth=out["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16),
        depth_conf=out["depth_conf"][0].float().cpu().numpy().astype(np.float16),
        input_hw=np.array(imgs.shape[-2:]), secs=secs, peak_mem=peak)
    meta["windows"][w["window_id"]] = dict(n_frames=len(idx), hw=list(imgs.shape[-2:]), secs=secs, peak_gib=peak / 2**30)
    print(w["window_id"], len(idx), tuple(imgs.shape[-2:]), f"{secs:.2f}s", f"{peak/2**30:.2f}GiB", flush=True)
    json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1)
json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1)
