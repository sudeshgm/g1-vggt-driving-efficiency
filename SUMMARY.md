
## Block 1 FINAL — accelerators vs frame drop (Oct 24 2026; Kaggle 2xT4 fp16, 11 cells, 57 min, quota 5h42/30h)
Tables: `results/block1_final/block1_final_tables.md` (+ final_main_table.csv, final_paired_deltas.csv, final_far_vs_near.csv);
plot `results/block1_final/acc_vs_flops_final.png`; script `scripts/block1_final.py`. Verified: 70 windows/cell, same
scored frames (0,4..28), 0 NaN, run_meta present. Decision set: turning excluded (weak), AUC without stationary; scene CIs.

Verdict:
1. **Beating frame drop at matched-or-lower FLOPs beyond the CI: no meaningful win.** Only GoToHunt (paper config)
   passes the mechanical test (N16 at 0.319 vs dense N16 0.364; N8 at 0.137 vs dense N8 0.148), and only on ATE
   (−5/−6 mm, CI barely excludes 0; one metric out of 4, with ~76 comparisons). AUC/PtMap/AbsRel are tied. It also
   misses its budget (it saves only 7-12% FLOPs). FastVGGT, AVGGT, Co-Me and GoToHunt-all are **worse** than plain
   frame drop on ≥2 metrics beyond the CI. Dense N8 (0.148) matches dense N32 within the CI, so frame drop is the strategy to beat, and nothing beats it.
2. **Far-field damage more than near-field: no.** For every cell, far−near relative AbsRel change vs the same-frame dense
   run has a CI that includes 0 or is negative (GoToHunt-all damages the near field more).
3. Co-Me T4 timing is INVALID (no sm80 kernels); `scripts/modal_come_a100.py` (A100, <16 min) measures fused vs unfused.
   It has not been run.
