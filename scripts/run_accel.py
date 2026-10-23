"""Accelerator runner. Same npz/run_meta schema as run_vggt.py -> scripts/score.py works unchanged.
Methods (official code vendored in third_party/, pinned commits in third_party/PINS.md):
  fastvggt  official FastVGGT package (merging=0, merge_ratio knob; paper default 0.9), stock VGGT-1B weights
  gotohunt  official GoToHunt attention on stock VGGT; frame/diverse selection OFF; paper config t=3, ds layers 2..8, gaf {0,1}
  gotohunt_all  same, but K/V downsampling in ALL non-gaf global layers (official standalone default) to reach the budget
  come      pure-PyTorch port of official Co-Me direct mode (3x3, ranking threshold knob, official predictor ckpt)
  avggt     our reimplementation (g1/avggt.py)
Budgets = FLOP fraction of dense N32 full-res (g1/flops_accel.py, method-specific, calculated). If a method cannot reach
a budget it is run at its closest achievable setting and flagged reachable=False (reported, not counted as meeting budget).
  python scripts/run_accel.py --plan
  python scripts/run_accel.py --method come --budget 0.5 --frames 32 --out preds/come_b0.5_N32 [--scale 0.5 --max_windows 1]
T4/sm75: fp16 autocast + SDPA. Co-Me fused kernels are sm80+; here the math-equivalent torch path is used (no speed claim)."""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from g1 import flops_accel as FA
BUDGETS = (0.5, 0.25, 0.1); T_EARLY = 9
CELLS = [("fastvggt", 0.5, 32), ("fastvggt", 0.25, 16), ("fastvggt", 0.1, 8), ("gotohunt", 0.5, 32), ("gotohunt", 0.25, 16),
         ("gotohunt", 0.1, 8), ("gotohunt_all", 0.5, 32), ("come", 0.5, 32), ("come", 0.25, 16), ("come", 0.1, 8), ("avggt", 0.25, 16)]
REPOS = dict(fastvggt="https://github.com/mystorm16/FastVGGT@6526e275a29572653a034762bb3c6c9ce280ff55",
             gotohunt="https://github.com/zsh2000/gotohunt@6c92c4f0c747847d85d78fadd3a426e7046e45dc",
             gotohunt_all="https://github.com/zsh2000/gotohunt@6c92c4f0c747847d85d78fadd3a426e7046e45dc (budget-forced all-layer variant)",
             come="https://github.com/co-me-tokens/CoMe@5c7cfbc14da37f5dbb991e3a6f9e2c4a5efb8833",
             avggt="(no official code; g1/avggt.py = our reimplementation)")

def solve(method, budget, n, scale=1.0):
    if method == "fastvggt":  # merge_ratio saturates (dst + 10% protected tokens never merge); paper default 0.9
        f = FA.fastvggt(n, scale, merge_ratio=0.9)["frac"]
        return dict(reachable=f <= budget, merge_ratio=0.9, merging=0, achieved=round(f, 4), note="" if f <= budget else f"min achievable {f:.3f} (merge saturates)")
    if method == "gotohunt":  # paper/README config: t=3, K/V downsampling in global layers 2..8, gaf {0,1}; floor ~0.82 at N32
        f = FA.gotohunt(n, scale, t=3, ds_layers=range(2, 9))["frac"]
        return dict(reachable=f <= budget, t=3, gaf_layers=[0, 1], ds_layers=list(range(2, 9)), achieved=round(f, 4),
                    note="" if f <= budget else f"paper config min {f:.3f}")
    if method == "gotohunt_all":  # budget-forced: official standalone path (ds in ALL non-gaf layers); smallest stride meeting budget
        fr = {t: FA.gotohunt(n, scale, t=t, ds_layers=range(2, 24))["frac"] for t in (2, 3, 4, 6, 8)}
        ok = [t for t, f in fr.items() if f <= budget]; t = min(ok) if ok else 8
        return dict(reachable=bool(ok), t=t, gaf_layers=[0, 1], ds_layers=list(range(2, 24)), achieved=round(fr[t], 4))
    if method == "come":  # smallest threshold quantile meeting budget (bisection)
        lo, hi = 0.0, 1.0
        if FA.come(n, scale, q=1.0)["frac"] > budget: return dict(reachable=False, threshold=1.0, achieved=round(FA.come(n, scale, q=1.0)["frac"], 4))
        for _ in range(30):
            mid = (lo + hi) / 2; lo, hi = (lo, mid) if FA.come(n, scale, q=mid)["frac"] <= budget else (mid, hi)
        q = round(hi + 5e-4, 3); return dict(reachable=True, threshold=q, encoder_layer=14, group=3, achieved=round(FA.come(n, scale, q=q)["frac"], 4))
    if method == "avggt":
        fr = {s: FA.avggt(n, scale, sigma=s, t_early=T_EARLY)["frac"] for s in (2, 4, 6, 9)}; ok = [s for s, f in fr.items() if f <= budget]
        s = min(ok) if ok else 9; return dict(reachable=bool(ok), sigma=s, t_early=T_EARLY, achieved=round(fr[s], 4))

def plan():
    return [dict(method=m, budget=b, frames=n, dense_frac=round(FA.dense(n)["frac"], 4), out=f"preds/{m}_b{b}_N{n}", **solve(m, b, n)) for m, b, n in CELLS]

def _weights():
    from huggingface_hub import hf_hub_download; import torch
    try:  # same file the controls used (model.safetensors); model.pt fallback
        from safetensors.torch import load_file; return load_file(hf_hub_download("facebook/VGGT-1B", "model.safetensors"))
    except Exception:
        return torch.load(hf_hub_download("facebook/VGGT-1B", "model.pt"), map_location="cpu", weights_only=True)

def load_avggt(cfg, dev, dt):
    from vggt.models.vggt import VGGT; from g1.avggt import apply_avggt
    m = VGGT(); m.load_state_dict(_weights()); apply_avggt(m.to(dev).eval(), sigma=cfg["sigma"], t_early=cfg["t_early"])
    return m, dict(impl="our reimplementation (g1/avggt.py)")
def load_fastvggt(cfg, dev, dt):
    import types, torch; from vggt.models.vggt import VGGT  # FastVGGT package (sys.path set in __main__)
    assert "FastVGGT" in sys.modules["vggt"].__file__, sys.modules["vggt"].__file__
    m = VGGT(merging=cfg["merging"], merge_ratio=cfg["merge_ratio"]); r = m.load_state_dict(_weights(), strict=False)
    m = m.to(dev).eval(); agg = m.aggregator; of = agg.forward
    def fwd(self, images):  # official attn defaults to a 28x37 grid; set the true grid of this input
        for b in self.global_blocks: b.attn.patch_height, b.attn.patch_width = images.shape[-2] // 14, images.shape[-1] // 14
        return of(images)
    agg.forward = types.MethodType(fwd, agg)
    return m, dict(impl="official FastVGGT (patched: dtype/cuda guards)", missing_keys=r.missing_keys,
                   unexpected_prefixes=sorted({k.split(".")[0] for k in r.unexpected_keys}))
def load_gotohunt(cfg, dev, dt):
    from vggt.models.vggt import VGGT; from g1.gotohunt import apply_gotohunt
    m = VGGT(); m.load_state_dict(_weights()); apply_gotohunt(m.to(dev).eval(), t=cfg["t"], gaf_layers=cfg["gaf_layers"], ds_layers=set(cfg["ds_layers"]))
    return m, dict(impl="official GoToHunt Attention.forward on stock VGGT, frame selection OFF")
def load_come(cfg, dev, dt):
    from vggt.models.vggt import VGGT; from g1.come import apply_come
    m = VGGT(); m.load_state_dict(_weights()); apply_come(m.to(dev).eval(), threshold=cfg["threshold"], encoder_layer=14, dtype=dt, device=dev)
    return m, dict(impl="pure-PyTorch port of official Co-Me direct mode (naive merge, SDPA per-key log-weight bias)")
ADAPTERS = dict(avggt=load_avggt, fastvggt=load_fastvggt, gotohunt=load_gotohunt, gotohunt_all=load_gotohunt, come=load_come)

def run(a, cfg):
    import time, numpy as np, torch, torch.nn.functional as F
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dev = "cuda" if torch.cuda.is_available() else "cpu"; dt = torch.float16 if dev == "cuda" else torch.float32
    os.environ.setdefault("FASTVGGT_DTYPE", "float16" if dev == "cuda" else "float32")
    model, extra = ADAPTERS[a.method](cfg, dev, dt); os.makedirs(a.out, exist_ok=True); stride = 32 // a.frames
    H, Wd = FA.come_hw(a.scale) if a.method == "come" else FA.input_hw(a.scale)
    meta = dict(method=a.method, budget=a.budget, frames=a.frames, scale=a.scale, plan=cfg, repo=REPOS[a.method], **extra,
                flops=FA.__dict__[a.method.replace("_all", "")](a.frames, a.scale, **{k: v for k, v in dict(merge_ratio=cfg.get("merge_ratio"), t=cfg.get("t"),
                      q=cfg.get("threshold"), sigma=cfg.get("sigma")).items() if v is not None and k in FA.__dict__[a.method.replace("_all", "")].__code__.co_varnames}),
                torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                capability=list(torch.cuda.get_device_capability(0)) if dev == "cuda" else None, dtype=str(dt), input_hw=[H, Wd], windows={})
    if a.method.startswith("gotohunt"): meta["flops"] = FA.gotohunt(a.frames, a.scale, t=cfg["t"], ds_layers=cfg["ds_layers"])
    meta["flops"] = {k: (float(v) if not isinstance(v, (list, tuple)) else v) for k, v in meta["flops"].items()}
    W = json.load(open(a.manifest))["windows"][: a.max_windows or None]
    for wi, w in enumerate(W):
        p = f"{a.out}/{w['window_id']}.npz"
        if os.path.exists(p): continue
        idx = list(range(0, 32, stride)); imgs = load_and_preprocess_images([os.path.join(a.img_root, w["frames"][i]["file"]) for i in idx]).to(dev)
        if tuple(imgs.shape[-2:]) != (H, Wd): imgs = F.interpolate(imgs, size=(H, Wd), mode="bilinear", antialias=True, align_corners=False)
        for _ in range(1 + (1 if (wi == 0 and dev == "cuda") else 0)):  # warm-up on first window (GPU)
            if dev == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad(), torch.autocast(dev, dtype=dt, enabled=dt != torch.float32): out = model(imgs)
            if dev == "cuda": torch.cuda.synchronize()
            secs = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() if dev == "cuda" else -1
        ext, intr = pose_encoding_to_extri_intri(out["pose_enc"], imgs.shape[-2:])
        np.savez_compressed(p, idx=np.array(idx), extrinsic=ext[0].float().cpu().numpy(), intrinsic=intr[0].float().cpu().numpy(),
            depth=out["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16), depth_conf=out["depth_conf"][0].float().cpu().numpy().astype(np.float16),
            input_hw=np.array(imgs.shape[-2:]), secs=secs, peak_mem=peak)
        wm = dict(n_frames=len(idx), hw=list(imgs.shape[-2:]), secs=secs, peak_gib=peak / 2**30)
        if a.method == "come": wm["merged_block_frac"] = model._come_state["merged_frac"]
        meta["windows"][w["window_id"]] = wm
        json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1, default=str); print(w["window_id"], f"{secs:.2f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", action="store_true")
    ap.add_argument("--method", choices=list(REPOS)); ap.add_argument("--budget", type=float, choices=BUDGETS)
    ap.add_argument("--frames", type=int, choices=(32, 16, 8), default=32); ap.add_argument("--out")
    ap.add_argument("--scale", type=float, default=1.0, help="input scale (CPU checks use 0.5; budgets are always solved at 1.0)")
    ap.add_argument("--manifest", default="manifests/windows.json"); ap.add_argument("--img_root", default="bundle/images")
    ap.add_argument("--max_windows", type=int, default=0); a = ap.parse_args()
    if a.plan:
        P = plan(); json.dump(P, open(os.path.join(ROOT, "manifests/accel_plan.json"), "w"), indent=1)
        for r in P: print(f"{r['method']:9s} b={r['budget']:<4} N{r['frames']:<2} dense={r['dense_frac']:.3f} -> {r['achieved']:.4f} "
                          f"{'OK' if r['reachable'] else 'NOT REACHABLE (run at min)'} " + str({k: r[k] for k in ('merge_ratio', 't', 'threshold', 'sigma') if k in r}))
        sys.exit()
    if a.method == "fastvggt": sys.path.insert(0, os.path.join(ROOT, "third_party", "FastVGGT"))
    run(a, solve(a.method, a.budget, a.frames))
