"""Score prediction dirs on the evaluation subset (idx%4==0) ONLY. Pose AUC@30 over all pairs of eval frames,
ATE (sim3), depth AbsRel/delta by LiDAR range band with coverage (median scale per window), 95% bootstrap CIs over
windows and over scenes (cluster bootstrap; windows in one scene are correlated). Usage: score.py preds/* --out results/block1"""
import argparse, json, os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1.metrics import *
ap = argparse.ArgumentParser(); ap.add_argument("preds", nargs="+"); ap.add_argument("--out", default="results/block1")
ap.add_argument("--manifest", default="manifests/windows.json"); a = ap.parse_args()
W = {w["window_id"]: w for w in json.load(open(a.manifest))["windows"]}
BANDS = ["0-15", "15-40", "40-80", "80-inf", "all"]; rows = []
for d in a.preds:
    for wid, w in W.items():
        f = f"{d}/{wid}.npz"
        if not os.path.exists(f): continue
        z = np.load(f); idx = list(z["idx"]); ev = w["eval_idx"]; assert set(ev) <= set(idx), "eval frame missing"
        pos = [idx.index(i) for i in ev]
        E = z["extrinsic"][pos]; pred = np.stack([np.linalg.inv(np.vstack([e, [0, 0, 0, 1]])) for e in E])
        gt = np.stack([np.array(w["frames"][i]["cam_to_world"]) for i in ev])
        auc = pose_auc(*pairwise_pose_errors(pred, gt)); a_te, (s, _, _) = ate(pred, gt)
        H, Wd = z["input_hw"]; P, G = [], []
        for k, i in zip(pos, ev):
            fr = w["frames"][i]
            if "depth_gt" not in fr: continue
            g = np.load(fr["depth_gt"]); dm = z["depth"][k].astype(np.float32)
            u = np.clip((g["u"] + .5) * Wd / 1600, 0, Wd - 1).astype(int); v = np.clip((g["v"] + .5) * H / 900, 0, H - 1).astype(int)
            P.append(dm[v, u]); G.append(g["d"].astype(np.float32))
        dmx = depth_metrics_binned(np.concatenate(P), np.concatenate(G))
        r = dict(config=os.path.basename(d), window=wid, scene=w["scene"], state=w["state"], auc30=auc, ate_m=a_te,
                 secs=float(z["secs"]), peak_gib=float(z["peak_mem"]) / 2**30)
        for b in BANDS: r[f"absrel_{b}"] = dmx[b]["absrel"]; r[f"d125_{b}"] = dmx[b]["d125"]; r[f"cov_{b}"] = dmx[b]["coverage"]
        rows.append(r)
os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/per_window.csv", "w") as fh:
    wr = csv.DictWriter(fh, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
summ = []
for c in sorted({r["config"] for r in rows}):
    R = [r for r in rows if r["config"] == c]
    for strat in ["all"] + sorted({r["state"] for r in R}):
        S = R if strat == "all" else [r for r in R if r["state"] == strat]
        o = dict(config=c, state=strat, n_windows=len(S), n_scenes=len({r["scene"] for r in S}))
        for m in ["auc30", "ate_m"] + [f"absrel_{b}" for b in BANDS] + [f"cov_{b}" for b in BANDS]:
            mu, lo, hi = bootstrap_ci([r[m] for r in S], n_boot=2000)
            sc = {}; [sc.setdefault(r["scene"], []).append(r[m]) for r in S]
            _, slo, shi = bootstrap_ci([np.nanmean(v) for v in sc.values()], n_boot=2000)
            o.update({m: mu, m + "_lo": lo, m + "_hi": hi, m + "_scene_lo": slo, m + "_scene_hi": shi})
        summ.append(o)
with open(f"{a.out}/summary.csv", "w") as fh:
    wr = csv.DictWriter(fh, fieldnames=summ[0].keys()); wr.writeheader(); wr.writerows(summ)
print(f"{len(rows)} rows -> {a.out}/summary.csv")
