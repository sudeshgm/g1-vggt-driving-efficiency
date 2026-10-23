"""Accelerator runner (SKELETON; finalise after GPU Bot's per-repo smoke tests). Same npz/run_meta schema as run_vggt.py,
so scripts/score.py works unchanged. Methods: fastvggt, gotohunt (diverse/frame selection OFF), come, avggt (our reimpl.).
Budgets are FLOP fractions of dense N32 full-res (g1/flops.py, calculated): main table 0.5/0.25/0.1 at N=32;
combo column = accelerator + frame drop (N16/N8). Unreachable cells -> "not reachable (min = x)" and are not run.
Baselines to beat (decision 5): 0.25 -> N16 (0.364 FLOPs is above budget; reported) / N8 (0.148), 0.1 -> N8.
  python scripts/run_accel.py --plan                       # CPU: print/emit plan (manifests/accel_plan.json)
  python scripts/run_accel.py --method avggt --budget 0.5 --frames 32 --out preds/avggt_b0.5_N32
T4/sm75: fp16 + SDPA (xformers fallback). Co-Me needs sm80+ kernels -> use its eager path on T4 or Modal A100 (one check)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1.flops import Cfg, ratio, solve_keep
from g1.avggt import avggt_kv_keep
BUDGETS = (0.5, 0.25, 0.1); FRAMES = (32, 16, 8); T_EARLY = 9; SIGMAS = (2, 4, 6, 9)  # paper grid sizes (g1/avggt.SIGMA_HW)
REPOS = dict(fastvggt="https://github.com/mystorm16/FastVGGT", gotohunt="https://github.com/zsh2000/gotohunt",
             come="https://github.com/co-me-tokens/CoMe", avggt="(no official code; g1/avggt.py = our reimplementation)")

def avggt_frac(n, sigma):
    """AVGGT FLOP fraction: layers < t_early as frame attention, later global layers keep sparse K/V grid."""
    from g1.flops import input_hw
    H, W = input_hw(); kv = avggt_kv_keep(n, (H // 14, W // 14), sigma)  # (21, 37) at full res
    return ratio(Cfg(n_frames=n, kv_keep_global=kv, n_global_as_frame=T_EARLY))

def solve(method, budget, n):
    """Return dict(param..., achieved) or dict(reachable=False, min=x). Frame drop alone <= budget -> flagged."""
    base = ratio(Cfg(n_frames=n))
    if base <= budget: return dict(reachable=False, note=f"frame drop alone already <= budget ({base:.3f})", min=base)
    if method == "avggt":
        fr = {s: avggt_frac(n, s) for s in SIGMAS}; ok = [s for s, f in fr.items() if f <= budget]
        if not ok: return dict(reachable=False, min=round(min(fr.values()), 3))
        s = min(ok); return dict(reachable=True, sigma=s, t_early=T_EARLY, achieved=round(fr[s], 4))
    # merging / pruning family: modelled as keep ratio of global-attention tokens (FastVGGT merge ratio = 1-keep;
    # GoToHunt token part, frame selection OFF; Co-Me merge ratio ~ 1-keep, confidence-guided). Exact repo knobs
    # are mapped in ADAPTERS after smoke test, achieved FLOPs re-counted from the actual kept-token schedule.
    x, ach = solve_keep(budget, n_frames=n)
    if x is None: return dict(reachable=False, min=round(ach, 3))
    return dict(reachable=True, keep_global=round(x, 4), merge_ratio=round(1 - x, 4), achieved=round(ach, 4))

def plan():
    P = []
    for m in REPOS:
        for b in BUDGETS:
            for n in FRAMES:
                r = solve(m, b, n); P.append(dict(method=m, budget=b, frames=n, column="main" if n == 32 else "combo",
                    out=f"preds/{m}_b{b}_N{n}", **r))
    return P

# ---------------- adapters (fill after smoke tests; each must return (model_callable, meta)) ----------------
def load_avggt(cfg, dev):
    from vggt.models.vggt import VGGT; from g1.avggt import apply_avggt
    m = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval(); apply_avggt(m, sigma=cfg["sigma"], t_early=cfg["t_early"])
    return m, dict(impl="our reimplementation (g1/avggt.py)")
def load_fastvggt(cfg, dev):
    # TODO(smoke): clone REPOS['fastvggt'] into third_party/FastVGGT; import its VGGT with merging=cfg['merge_ratio'];
    # confirm kwarg name + which layers merge; recount FLOPs from its schedule; pure PyTorch -> T4 OK.
    raise NotImplementedError("FastVGGT adapter pending smoke test")
def load_gotohunt(cfg, dev):
    # TODO(smoke): MapAnything env; set --token_ds_layers/--global_as_frame_layers/--frame_token_downsample to hit
    # cfg['achieved']; FORCE frame/diverse selection OFF (--frame_strategy none) so scored frames are identical.
    raise NotImplementedError("GoToHunt adapter pending smoke test")
def load_come(cfg, dev):
    # TODO(smoke): CoMe custom CUDA kernels effectively sm80+; on T4 use eager/unfused path if it exists (accuracy only),
    # wall-clock from Modal A100 single check. Merge ratio ~ cfg['merge_ratio'].
    raise NotImplementedError("Co-Me adapter pending smoke test")
ADAPTERS = dict(avggt=load_avggt, fastvggt=load_fastvggt, gotohunt=load_gotohunt, come=load_come)

def run(a, cfg):
    import time, numpy as np, torch
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    dev = "cuda" if torch.cuda.is_available() else "cpu"; dt = torch.float16 if dev == "cuda" else torch.float32
    model, extra = ADAPTERS[a.method](cfg, dev); os.makedirs(a.out, exist_ok=True); stride = 32 // a.frames
    meta = dict(method=a.method, budget=a.budget, frames=a.frames, plan=cfg, **extra, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                capability=list(torch.cuda.get_device_capability(0)) if dev == "cuda" else None, dtype=str(dt), windows={})
    W = json.load(open(a.manifest))["windows"][: a.max_windows or None]
    for wi, w in enumerate(W):
        p = f"{a.out}/{w['window_id']}.npz"
        if os.path.exists(p): continue
        idx = list(range(0, 32, stride)); imgs = load_and_preprocess_images([os.path.join(a.img_root, w["frames"][i]["file"]) for i in idx]).to(dev)
        for _ in range(1 + (1 if wi == 0 else 0)):  # warm-up on first window
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
        meta["windows"][w["window_id"]] = dict(n_frames=len(idx), hw=list(imgs.shape[-2:]), secs=secs, peak_gib=peak / 2**30)
        json.dump(meta, open(f"{a.out}/run_meta.json", "w"), indent=1); print(w["window_id"], f"{secs:.2f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--plan", action="store_true")
    ap.add_argument("--method", choices=list(REPOS)); ap.add_argument("--budget", type=float, choices=BUDGETS)
    ap.add_argument("--frames", type=int, choices=FRAMES, default=32); ap.add_argument("--out")
    ap.add_argument("--manifest", default="manifests/windows.json"); ap.add_argument("--img_root", default="bundle/images")
    ap.add_argument("--max_windows", type=int, default=0); a = ap.parse_args()
    if a.plan:
        P = plan(); json.dump(P, open("manifests/accel_plan.json", "w"), indent=1)
        for r in P: print(f"{r['method']:9s} b={r['budget']:<4} N{r['frames']:<2} {r['column']:5s} " + (f"RUN {({k: v for k, v in r.items() if k in ('sigma','keep_global','achieved')})}" if r["reachable"] else f"not reachable (min = {r['min']:.3f}) {r.get('note','')}"))
        sys.exit()
    cfg = solve(a.method, a.budget, a.frames)
    if not cfg["reachable"]: sys.exit(f"not reachable (min = {cfg['min']:.3f}) {cfg.get('note','')}")
    run(a, cfg)
