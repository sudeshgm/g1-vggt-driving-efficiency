"""Analytic FLOP tables (fraction of dense VGGT-1B, N=32, 294x518). Main table = method alone; combos = separate."""
import csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1.flops import *
from g1.avggt import avggt_kv_keep
ref = vggt_flops(Cfg()); out = "results/block1"; rows = []
def row(table, name, c, budget="", note=""):
    f = vggt_flops(c); rows.append(dict(table=table, config=name, budget=budget, n_frames=c.n_frames, H=f["H"], W=f["W"],
        keep_global=round(c.keep_global, 4), kv_keep=round(c.kv_keep_global, 4), global_as_frame=c.n_global_as_frame,
        TFLOPs=round(f["total"] / 1e12, 2), frac_dense=round(f["total"] / ref["total"], 4), note=note))
for n in (32, 16, 8): row("controls", f"dense_N{n}", Cfg(n))
for s in (0.707, 0.5, 0.354, 0.25): row("controls", f"dense_res{s}", Cfg(scale=s))
for sg in (2, 4, 6, 9):
    for n in (32, 16, 8):
        kv = avggt_kv_keep(n, (21, 37), sg)
        row("avggt_native" if n == 32 else "combo_framedrop", f"avggt_s{sg}_t9_N{n}", Cfg(n, kv_keep_global=kv, n_global_as_frame=9),
            note="our reimplementation; analytic")
floor = ratio(Cfg(keep_global=0.0)); avg_floor = ratio(Cfg(kv_keep_global=avggt_kv_keep(32, (21, 37), 9), n_global_as_frame=9))
for b in (0.5, 0.25, 0.1):
    k, r = solve_keep(b)
    row("main", f"globaltoken_keep_b{b}", Cfg(keep_global=k or 0.0), b,
        "" if k is not None else f"not reachable (min = {floor:.3f})")
    row("main", f"avggt_b{b}", Cfg(kv_keep_global=avggt_kv_keep(32, (21, 37), 9), n_global_as_frame=9), b,
        "use sigma giving <= budget" if avg_floor <= b else f"not reachable (min = {avg_floor:.3f} at sigma=9,t_early=9)")
    for n in (16, 8):
        k, r = solve_keep(b, n_frames=n)
        if ratio(Cfg(n)) <= b: row("combo_framedrop", f"globaltoken_keep_N{n}_b{b}", Cfg(n), b, f"frame-drop alone already <= budget ({ratio(Cfg(n)):.3f}); no merge needed")
        elif k is not None: row("combo_framedrop", f"globaltoken_keep_N{n}_b{b}", Cfg(n, keep_global=k), b, "combo column only")
        else: row("combo_framedrop", f"globaltoken_keep_N{n}_b{b}", Cfg(n, keep_global=0.0), b,
                  f"not reachable (min = {ratio(Cfg(n, keep_global=0.0)):.3f})")
w = csv.DictWriter(open(f"{out}/flop_table.csv", "w"), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for r in rows: print(r["table"], r["config"], r["budget"], r["frac_dense"], "keep=", r["keep_global"], r["note"])
