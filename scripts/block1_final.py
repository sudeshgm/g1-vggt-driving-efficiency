"""Block-1 FINAL table: accelerators vs frame-drop baselines (pre-registered rules).
Scored frames 0,4..28; >80 m reported not scored; scene-cluster bootstrap CIs (primary); decision set = all windows except
turning (WEAK, excluded); AUC@30 additionally excludes stationary windows (AUC undefined). Inputs: results/block1_final/
per_window.csv (score.py over dense controls + accel preds) and run_meta.json under argv[1]."""
import csv, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1.metrics import bootstrap_ci
from g1.flops import ratio, Cfg
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
O = "results/block1_final"; META = sys.argv[1] if len(sys.argv) > 1 else "/workspace/g1_preds/preds"
pw = list(csv.DictReader(open(f"{O}/per_window.csv"))); NB = 4000
DENSE = {"dense_N32": ratio(Cfg(32, 1.)), "dense_N16": ratio(Cfg(16, 1.)), "dense_N8": ratio(Cfg(8, 1.)), "dense_res0.5": ratio(Cfg(32, .5)), "dense_res0.25": ratio(Cfg(32, .25))}
ACC = sorted({r["config"] for r in pw} - set(DENSE)); meta = {}
for c in list(DENSE) + ACC:
    p = f"{META}/{c}/run_meta.json" if c in DENSE else f"{META}/accel_full/{c}/run_meta.json"; meta[c] = json.load(open(p))
FR = dict(DENSE); FR.update({c: meta[c]["flops"]["frac"] for c in ACC})
BUD = {c: float(c.split("_b")[1].split("_")[0]) for c in ACC}; NF = {c: int(c.split("_N")[1]) for c in ACC}
METH = {c: c.split("_b")[0] for c in ACC}
MET = [("auc30", "AUC@30", +1), ("ate_m", "ATE (m)", -1), ("pm_rel_all", "PtMap rel", -1), ("absrel_40-80", "AbsRel 40-80", -1)]
by = {(r["config"], r["window"]): r for r in pw}; WIN = sorted({r["window"] for r in pw})
info = {r["window"]: (r["scene"], r["state"]) for r in pw}
def wins(m): return [w for w in WIN if info[w][1] != "turning" and not (m == "auc30" and info[w][1] == "stationary")]
def val(c, w, m): return float(by[(c, w)][m])
def scene_mean_ci(c, m):
    sc = {}; [sc.setdefault(info[w][0], []).append(val(c, w, m)) for w in wins(m)]
    return bootstrap_ci([np.nanmean(v) for v in sc.values()], n_boot=NB)
def paired(c, b, m, rel=False):
    sc = {}
    for w in wins(m):
        d = val(c, w, m) - val(b, w, m); d = d / val(b, w, m) if rel else d; sc.setdefault(info[w][0], []).append(d)
    return bootstrap_ci([np.nanmean(v) for v in sc.values()], n_boot=NB)
def secs(c):
    s = [v["secs"] for v in meta[c]["windows"].values()]; return float(np.median(s[1:]))
def sig(mu, lo, hi, d):  # +: better beyond CI, -: worse beyond CI, 0: CI includes 0
    if lo > 0: return "+" if d > 0 else "-"
    if hi < 0: return "-" if d > 0 else "+"
    return "0"
def f3(x): return f"{x:.3f}"
L = ["# Block 1 FINAL — accelerators vs frame drop (Kaggle T4, fp16; 70 windows / 10 scenes)", "",
     "Rules: frames 0,4..28 scored; >80 m not scored; 95% scene-cluster bootstrap CIs; decision set excludes turning windows "
     "(n=9, weak); AUC@30 also excludes stationary (n=18) -> AUC n=43 windows, other metrics n=61. FLOP fractions are calculated "
     "(analytic, vs dense N32 full-res). T4 s/window = median excl. first window. Co-Me T4 timing INVALID (math-equivalent "
     "PyTorch path, no fused sm80+ kernels).", "",
     "## Main table (value [scene CI], decision set)", "",
     "| config | budget | frames | FLOP frac | met | " + " | ".join(m[1] for m in MET) + " | T4 s/window |", "|" + "---|" * (6 + len(MET))]
order = list(DENSE) + sorted(ACC, key=lambda c: (-BUD[c], METH[c]))
rows = []
for c in order:
    vals = {m: scene_mean_ci(c, m) for m, _, _ in MET}
    b = BUD.get(c); met = "—" if b is None else ("met" if FR[c] <= b + 1e-9 else "MISSED")
    t = secs(c); ts = f"{t:.2f}" + (" (INVALID)" if c.startswith("come") else "")
    L.append(f"| {c} | {b if b else '—'} | {NF.get(c, '')} | {FR[c]:.3f} | {met} | " + " | ".join(f"{f3(v[0])} [{f3(v[1])}, {f3(v[2])}]" for v in vals.values()) + f" | {ts} |")
    rows.append(dict(config=c, budget=b, frames=NF.get(c), flop_frac=FR[c], met=met, t4_s_per_window=t, t4_timing_valid=not c.startswith("come"),
                     **{m: v[0] for m, v in vals.items()}, **{m + "_lo": v[1] for m, v in vals.items()}, **{m + "_hi": v[2] for m, v in vals.items()}))
with open(f"{O}/final_main_table.csv", "w") as fh: w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)
L += ["", "Combo column = the N16/N8 cells (accelerator + frame drop); main column at N32 is the 0.5 cells.", "",
      "## Paired deltas vs frame-drop baselines (accelerator − baseline, scene CI; sign: + better beyond CI, − worse, 0 within CI)", "",
      "| accel cell | baseline | FLOP accel / base | matched-or-lower FLOPs | " + " | ".join(f"Δ {m[1]}" for m in MET) + " |", "|" + "---|" * (4 + len(MET))]
drows = []; verdict_beat = []
for c in sorted(ACC, key=lambda c: (-BUD[c], METH[c])):
    for b in (["dense_N16", "dense_N8"] if BUD[c] >= 0.25 else ["dense_N8"]):
        cells = []; signs = []
        for m, _, d in MET:
            mu, lo, hi = paired(c, b, m); s = sig(mu, lo, hi, d); signs.append(s)
            cells.append(f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}] {s}")
            drows.append(dict(config=c, baseline=b, metric=m, delta=mu, lo=lo, hi=hi, sign=s, flop_accel=FR[c], flop_base=FR[b]))
        ok = FR[c] <= FR[b] + 1e-9
        if ok and "+" in signs and "-" not in signs: verdict_beat.append((c, b, signs))
        L.append(f"| {c} | {b} | {FR[c]:.3f} / {FR[b]:.3f} | {'yes' if ok else 'no'} | " + " | ".join(cells) + " |")
with open(f"{O}/final_paired_deltas.csv", "w") as fh: w = csv.DictWriter(fh, drows[0].keys()); w.writeheader(); w.writerows(drows)
# far vs near: relative AbsRel change vs dense at the SAME frame count (isolates the accelerator), far minus near
L += ["", "## Far-field vs near-field damage (relative AbsRel change vs dense at same frame count; 40-80 m minus 0-15 m; scene CI)", "",
      "| accel cell | ref | rel Δ AbsRel 0-15 | rel Δ AbsRel 40-80 | far − near | far worse beyond CI? |", "|---|---|---|---|---|---|"]
far = []; frows = []
for c in sorted(ACC, key=lambda c: (-BUD[c], METH[c])):
    ref = f"dense_N{NF[c]}"; n = paired(c, ref, "absrel_0-15", True); fa = paired(c, ref, "absrel_40-80", True)
    sc = {}
    for w in wins("absrel_0-15"):
        d = (val(c, w, "absrel_40-80") / val(ref, w, "absrel_40-80") - 1) - (val(c, w, "absrel_0-15") / val(ref, w, "absrel_0-15") - 1)
        sc.setdefault(info[w][0], []).append(d)
    fn = bootstrap_ci([np.nanmean(v) for v in sc.values()], n_boot=NB); worse = fn[1] > 0
    if worse: far.append(c)
    frows.append(dict(config=c, ref=ref, rel_near=n[0], rel_far=fa[0], far_minus_near=fn[0], lo=fn[1], hi=fn[2], far_worse=worse))
    L.append(f"| {c} | {ref} | {n[0]:+.1%} [{n[1]:+.1%}, {n[2]:+.1%}] | {fa[0]:+.1%} [{fa[1]:+.1%}, {fa[2]:+.1%}] | {fn[0]:+.1%} [{fn[1]:+.1%}, {fn[2]:+.1%}] | {'YES' if worse else 'no'} |")
with open(f"{O}/final_far_vs_near.csv", "w") as fh: w = csv.DictWriter(fh, frows[0].keys()); w.writeheader(); w.writerows(frows)
L += ["", "## Decision", "",
      "Q1 — does any accelerator beat frame drop at matched-or-lower FLOPs by more than the CI (≥1 metric better beyond CI, none worse)? "
      + ("**YES**: " + "; ".join(f"{c} vs {b}" for c, b, _ in verdict_beat) if verdict_beat else "**NO.**"),
      "", "Q2 — does any accelerator damage the far field (40-80 m) more than the near field (0-15 m) beyond the CI? "
      + ("**YES**: " + ", ".join(far) if far else "**NO.**")]
open(f"{O}/block1_final_tables.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
# plot
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8)); mk = dict(fastvggt="s", gotohunt="^", gotohunt_all="v", come="D", avggt="P"); col = dict(fastvggt="tab:purple", gotohunt="tab:olive", gotohunt_all="tab:gray", come="tab:orange", avggt="tab:blue")
for a_, (m, lab, _) in zip(ax, [MET[0], MET[3], MET[2]]):
    xs = [FR[c] for c in ["dense_N8", "dense_N16", "dense_N32"]]; ys = [scene_mean_ci(c, m)[0] for c in ["dense_N8", "dense_N16", "dense_N32"]]
    a_.plot(xs, ys, "k-o", label="frame drop (N8/N16/N32)")
    for c in ["dense_res0.5", "dense_res0.25"]: a_.plot(FR[c], scene_mean_ci(c, m)[0], "kx", ms=9); a_.annotate(c[6:], (FR[c], scene_mean_ci(c, m)[0]), fontsize=7)
    for c in ACC:
        mu, lo, hi = scene_mean_ci(c, m)
        a_.errorbar(FR[c], mu, yerr=[[mu - lo], [hi - mu]], fmt=mk[METH[c]], color=col[METH[c]], ms=7, capsize=2, label=f"{METH[c]}" if c == min(x for x in ACC if METH[x] == METH[c]) else None)
        a_.annotate(f"N{NF[c]}", (FR[c], mu), fontsize=7, xytext=(3, 3), textcoords="offset points")
    for b in (0.1, 0.25, 0.5): a_.axvline(b, color="0.85", ls="--", lw=.8)
    a_.set_xscale("log"); a_.set_xlabel("FLOP fraction of dense N32 (calculated)"); a_.set_ylabel(lab + (" (higher better)" if m == "auc30" else " (lower better)")); a_.set_title(lab)
ax[0].legend(fontsize=7); fig.suptitle("G1 Block 1: accuracy vs FLOPs (decision set, scene CIs; T4 fp16, 70 windows)"); fig.tight_layout()
fig.savefig(f"{O}/acc_vs_flops_final.png", dpi=130); print("plot ->", f"{O}/acc_vs_flops_final.png")
