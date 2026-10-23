"""Block-1 controls: verification + results table + motion breakdown + accuracy-vs-FLOP plot.
Inputs: results/block1/{per_window,summary}.csv (score.py), flop_table.csv, preds/*/run_meta.json (T4)."""
import csv, json, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1.flops import vggt_flops, ratio, Cfg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = "results/block1"; META = sys.argv[1] if len(sys.argv) > 1 else "/workspace/g1_preds/preds"
CF = {"dense_N32": Cfg(32, 1.0), "dense_N16": Cfg(16, 1.0), "dense_N8": Cfg(8, 1.0), "dense_res0.5": Cfg(32, .5), "dense_res0.25": Cfg(32, .25)}
LBL = {"dense_N32": "N32 (dense)", "dense_N16": "N16", "dense_N8": "N8", "dense_res0.5": "res0.5", "dense_res0.25": "res0.25"}
base = Cfg(32, 1.0); FR = {c: ratio(v, base) for c, v in CF.items()}
pw = list(csv.DictReader(open(f"{O}/per_window.csv"))); sm = list(csv.DictReader(open(f"{O}/summary.csv")))
# ---------- verification ----------
V = []
for c in CF:
    R = [r for r in pw if r["config"] == c]; V.append(f"{c}: {len(R)} windows")
    for k in R[0]:
        if k in ("config", "window", "scene", "state") or "80-inf" in k: continue
        nn = sum(not np.isfinite(float(r[k])) for r in R)
        if nn: V.append(f"  NaN {k}: {nn}")
win = {c: sorted(r["window"] for r in pw if r["config"] == c) for c in CF}
V.append("identical window sets: " + str(len({tuple(v) for v in win.values()}) == 1))
nn = {c: [r["n_all"] for r in sorted((r for r in pw if r["config"] == c), key=lambda r: r["window"])] for c in CF}
V.append("identical scored LiDAR px per window across configs: " + str(len({tuple(v) for v in nn.values()}) == 1))
meta = {}
for c in list(CF) + ["dense_N32_timingrep"]:
    m = json.load(open(f"{META}/{c}/run_meta.json")); w = m["windows"]; meta[c] = m
    V.append(f"{c}: gpu={m['gpu']} cc={m['capability']} torch={m['torch']} cuda={m['cuda']} dtype={m['dtype']} attn_flash={m.get('sdpa_flash_enabled')} n={len(w)} frames={sorted({v['n_frames'] for v in w.values()})} hw={sorted({tuple(v['hw']) for v in w.values()})}")
open(f"{O}/verification.txt", "w").write("\n".join(V) + "\n"); print("\n".join(V))
def wall(c):
    s = np.array([v["secs"] for v in meta[c]["windows"].values()]); s = s[1:]  # drop first (warm-up/compile)
    return float(np.median(s)), float(max(v["peak_gib"] for v in meta[c]["windows"].values()))
rep = np.median([v["secs"] for v in meta["dense_N32_timingrep"]["windows"].values()])
# ---------- tables ----------
M = [("auc30", "AUC@30", 3), ("ate_m", "ATE m", 3), ("pm_rel_all", "PtMap rel", 3), ("pm_rel_40-80", "PtMap rel 40-80", 3),
     ("absrel_0-15", "AbsRel 0-15", 3), ("absrel_15-40", "AbsRel 15-40", 3), ("absrel_40-80", "AbsRel 40-80", 3), ("absrel_all", "AbsRel all", 3)]
def cell(o, k, p): return f"{float(o[k]):.{p}f} [{float(o[k+'_scene_lo']):.{p}f}, {float(o[k+'_scene_hi']):.{p}f}]"
def get(c, st): return next((o for o in sm if o["config"] == c and o["state"] == st), None)
rows, md = [], []
hdr = ["config", "FLOP frac"] + [m[1] for m in M] + ["cov >80 m", "T4 s/window (median)", "speed-up (calc.)", "peak GiB"]
md.append("| " + " | ".join(hdr) + " |"); md.append("|" + "---|" * len(hdr))
for c in CF:
    o = get(c, "all"); s, pk = wall(c)
    r = dict(config=c, flop_frac=round(FR[c], 4), **{m[0]: float(o[m[0]]) for m in M}, **{m[0] + "_lo": float(o[m[0] + "_scene_lo"]) for m in M},
             **{m[0] + "_hi": float(o[m[0] + "_scene_hi"]) for m in M}, cov_80inf=float(o["cov_80-inf"]), t4_secs=s, speedup_calc=1 / FR[c], peak_gib=pk)
    rows.append(r)
    md.append("| " + " | ".join([LBL[c], f"{FR[c]:.3f}"] + [cell(o, m[0], m[2]) for m in M] + [f"{float(o['cov_80-inf']):.3f}", f"{s:.2f}", f"{1/FR[c]:.1f}x", f"{pk:.2f}"]) + " |")
with open(f"{O}/block1_controls_table.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
# motion breakdown
mb = ["| config | state | n win (scenes) | " + " | ".join(m[1] for m in M[:1] + M[2:4] + M[4:7]) + " | ATE m |", "|" + "---|" * 10]
mrows = []
for st in ["stationary", "straight", "turning", "other"]:
    for c in CF:
        o = get(c, st)
        if not o: continue
        mrows.append({"config": c, "state": st, "in_decision": st not in ("turning",), "auc_reported": st != "stationary", "n_windows": o["n_windows"], "n_scenes": o["n_scenes"], **{m[0]: float(o[m[0]]) for m in M}})
        cs_ = ["n/a (pose AUC undefined when stationary)" if (st == "stationary" and m[0] == "auc30") else cell(o, m[0], 3) for m in M[:1] + M[2:7]]
        tag = {"turning": "turning (WEAK: n=9/4 scenes; excluded from decision)", "stationary": "stationary (ATE+depth only)"}.get(st, st)
        mb.append(f"| {LBL[c]} | {tag} | {o['n_windows']} ({o['n_scenes']}) | " + " | ".join(cs_) + f" | {float(o['ate_m']):.3f} |")
with open(f"{O}/block1_motion_breakdown.csv", "w") as fh:
    w = csv.DictWriter(fh, fieldnames=mrows[0].keys()); w.writeheader(); w.writerows(mrows)
dl = list(csv.DictReader(open(f"{O}/delta_vs_dense.csv")))
dmd = ["| config | metric | delta vs N32 | scene 95% CI |", "|---|---|---|---|"] + [f"| {LBL.get(d['config'], d['config'])} | {d['metric']} | {float(d['delta_vs_dense_mean']):+.4f} | [{float(d['scene_ci_lo']):+.4f}, {float(d['scene_ci_hi']):+.4f}] |" for d in dl if d["config"] != "dense_N32"]
S2 = "results/block1/stride2_check/delta_vs_dense.csv"
s2md = ""
if os.path.exists(S2):
    s2md = "\n\n## Extra check (NOT stop rule): N16 vs N32 on frames 0,2,...,30 (scene CI)\nNote: these are exactly N16's input frames, so this tests more scored frames, not recovery of skipped frames.\n\n| metric | delta | scene 95% CI |\n|---|---|---|\n" + "\n".join(f"| {d['metric']} | {float(d['delta_vs_dense_mean']):+.4f} | [{float(d['scene_ci_lo']):+.4f}, {float(d['scene_ci_hi']):+.4f}] |" for d in csv.DictReader(open(S2)) if d["config"] == "dense_N16")
open(f"{O}/block1_tables.md", "w").write("## Main (all 70 windows / 10 scenes; scene-level 95% CI)\n\n" + "\n".join(md) +
    f"\n\nTiming repeat N32 (10 windows): median {rep:.2f} s/window.\n\n## Delta vs dense N32 (paired, scene CI)\n\n" + "\n".join(dmd) +
    "\n\n## By motion label (decision uses all/straight/other/stationary; turning reported, weak, excluded; stationary: no AUC)\n\n" + "\n".join(mb) + s2md + "\n")
# ---------- plot ----------
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a_, (k, t) in zip(ax, [("auc30", "Pose AUC@30 (higher better)"), ("absrel_40-80", "AbsRel 40-80 m (lower better)"), ("absrel_0-15", "AbsRel 0-15 m / PtMap rel all")]):
    for fam, cs, mk in [("frame drop", ["dense_N32", "dense_N16", "dense_N8"], "o-"), ("resolution", ["dense_N32", "dense_res0.5", "dense_res0.25"], "s--")]:
        R = [r for c in cs for r in rows if r["config"] == c]
        a_.errorbar([r["flop_frac"] for r in R], [r[k] for r in R], yerr=[[r[k] - r[k + "_lo"] for r in R], [r[k + "_hi"] - r[k] for r in R]], fmt=mk, capsize=3, label=fam)
        if k == "absrel_0-15": a_.plot([r["flop_frac"] for r in R], [r["pm_rel_all"] for r in R], mk[0] + ":", alpha=.5, label=fam + " (PtMap rel)")
    for r in rows: a_.annotate(LBL[r["config"]].split()[0], (r["flop_frac"], r[k]), fontsize=7)
    a_.set_xscale("log"); a_.set_xlabel("FLOP fraction of dense N32 (calculated)"); a_.set_title(t); a_.grid(alpha=.3)
ax[0].legend(fontsize=7); ax[2].legend(fontsize=7)
fig.suptitle("G1 Block 1 controls - VGGT-1B fp16 on T4, nuScenes-mini CAM_FRONT 32-frame windows (scene 95% CI)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{O}/acc_vs_flops.png", dpi=130); print(open(f"{O}/block1_tables.md").read())
