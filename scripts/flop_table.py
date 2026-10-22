import csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from g1.flops import *
rows = []; ref = vggt_flops(Cfg())
def add(name, c, note=""):
    f = vggt_flops(c); rows.append(dict(config=name, n_frames=c.n_frames, H=f["H"], W=f["W"], patches=f["patches"],
        keep_global=round(c.keep_global, 4), kv_keep=round(c.kv_keep_global, 4), global_as_frame=c.n_global_as_frame,
        TFLOPs=round(f["total"] / 1e12, 2), global_share=round(f["global_"] / f["total"], 3), frac_dense=round(f["total"] / ref["total"], 4), note=note))
add("dense_N32", Cfg()); add("dense_N16", Cfg(16)); add("dense_N8", Cfg(8))
for s in (0.707, 0.5, 0.354, 0.25): add(f"dense_res{s}", Cfg(scale=s))
floor = ratio(Cfg(keep_global=0.0))
for b in (0.5, 0.25, 0.1):
    k, r = solve_keep(b)
    add(f"tokenmerge_b{b}", Cfg(keep_global=k if k is not None else 0.0), "" if k is not None else f"UNREACHABLE: global-only reduction floor={floor:.3f}")
    k, r = solve_keep(b, field="kv_keep_global", n_global_as_frame=12)
    add(f"avggt_g12_b{b}", Cfg(kv_keep_global=k if k is not None else 0.0, n_global_as_frame=12), "" if k is not None else "UNREACHABLE")
w = csv.DictWriter(open("results/block1/flop_table.csv", "w"), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
print("ref TFLOPs", ref["total"]/1e12, "parts", {k: round(v/ref['total'],3) for k,v in ref.items() if k in ('encoder','frame','global_','heads_est')})
