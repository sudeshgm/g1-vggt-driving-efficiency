# Block 1 (CPU prep) summary — 2026-10-22
- Windows: 70 non-overlapping 32-frame CAM_FRONT 12 Hz sweep windows from 10 nuScenes-mini scenes (0 skipped; 7/scene max).
  States (data/NOTES.md thresholds): straight 30, stationary 18, turning 9, other 13 (slow or mild curve; not forced into a class).
  Only 10 scenes -> report scene-cluster bootstrap CIs alongside window CIs; turning has n=9 from 4 scenes (weak).
- Eval subset: idx 0,4,...,28 (8 frames/window) — present in N, N/2, N/4. 550/560 eval frames have LiDAR within 30 ms (median 14.6 ms), ego-motion compensated.
- LiDAR pixel share by band: 0-15 m 57.1 %, 15-40 m 33.6 %, 40-80 m 9.0 %, >80 m 0.3 % (~9 px/frame) -> the >80 m band is essentially unmeasurable with nuScenes 32-beam LiDAR.
- Tests: 19/19 pass (test_results.txt). Overlay: overlay_scene-0061_w0_f4.png.
- FLOPs: flop_table.csv. Dense N32 = 113.4 TFLOP; N16 0.364, N8 0.148, res0.707 0.361, res0.5 0.133, res0.354 0.061. Global-only token reduction floor = 0.324.
- No GPU results yet. See ../../GPU_PLAN.md.

## Update 07:20 IST — decisions applied
- AVGGT reimplemented (g1/avggt.py, "our reimplementation"); tests: SGA == explicit reference, sigma=1/t_early=0 == dense VGGT exactly.
  Faithful AVGGT (t_early=9) at N=32: sigma 2/4/6/9 -> 0.644/0.564/0.536/0.515 of dense => main-table 0.5/0.25/0.1 all "not reachable (min = 0.515)".
- Global-token reduction: 0.5 reachable (keep 0.46); 0.25/0.1 not reachable (min = 0.324). Combo column in flop_table.csv.
- >80 m reported (coverage/n) not scored; scene-level CIs primary; delta_vs_dense.csv added.
- CPU dry-run with REAL VGGT-1B weights (fp32, 2 windows, N/4 frames, 0.5 res): 5.2/6.1 s per window; AUC@30 0.615/0.620,
  ATE 0.10/0.17 m, AbsRel 0-15 m 0.10/0.13, 40-80 m 0.31/0.38 (dry-run only, not a result).

## Update 2026-10-23 — GPU controls returned (Kaggle 2x Tesla T4 sm75, torch 2.11.0+cu128, fp16 SDPA; 28m44s, exit 0)
Tables: block1_tables.md / block1_controls_table.csv / block1_motion_breakdown.csv; plot: acc_vs_flops.png; checks: verification.txt.
Re-scored from preds (adds point-map error = mean ||p-g||/d_gt after Umeyama sim(3) of pred depth+K+pose points onto LiDAR world points);
AUC/ATE/AbsRel identical to the Kaggle-side scoring (max diff 7e-14). 70 windows x 5 configs, same windows and same LiDAR px; no NaNs in scored columns.

| config | FLOP | AUC@30 | ATE m | AbsRel 40-80 | PtMap rel 40-80 | AbsRel all | T4 s/win | peak GiB |
|---|---|---|---|---|---|---|---|---|
| N32 | 1.000 | 0.728 | 0.097 | 0.248 | 0.202 | 0.122 | 12.35 | 8.94 |
| N16 | 0.364 | 0.729 | 0.099 | 0.248 | 0.201 | 0.121 | 4.64 | 8.39 |
| N8 | 0.148 | 0.729 | 0.098 | 0.249 | 0.200 | 0.121 | 2.06 | 8.11 |
| res0.5 | 0.133 | 0.481 | 0.199 | 0.374 | 0.300 | 0.197 | 1.96 | 6.99 |
| res0.25 | 0.030 | 0.348 | 2.094 | 0.601 | 0.443 | 0.382 | 0.61 | 7.11 |

**Finding.** At about 0.15 of the FLOPs, dropping frames (N8) loses nothing measurable. Every paired delta vs N32 is under 0.003, and every scene CI includes 0. For example, AbsRel 40-80 m changes by +0.001 [-0.002, +0.004] and AUC@30 by +0.001 [-0.001, +0.005]. At a similar 0.13, halving the resolution is much worse. AUC@30 drops by 0.247 [0.168, 0.313], AbsRel 40-80 m rises by +0.126 [+0.098, +0.159], and ATE doubles. The far-field band suffers most in relative terms (+51% vs +73% near; all bands degrade). res0.25 breaks down: ATE is 2.1 m.
So the cheap frame-drop control keeps far-field depth and pose, and an accelerator has to beat N8, not N32.

**Caveats.**
- (1) Scoring uses only frames 0,4,…,28, which are exactly the N8 input frames. N8 is therefore never asked to recover frames it skipped, and frame dropping is favoured. Scoring all 32 frames would need interpolated poses and depth for the skipped frames.
- (2) In stationary windows (18 windows, 3 scenes), AUC@30 is about 0.09 for every config. The direction of near-zero translation is noise, so the translation-angle term is undefined. AUC for stationary windows is not meaningful; ATE and depth are fine.
- (3) The calculated FLOP speed-up and the T4 wall-clock speed-up differ. N8 is 6.0x faster on the T4 against 6.7x calculated. res0.5 is 6.3x against 7.5x, and res0.25 is 20x against 33.7x.
- (4) Peak memory is 7–9 GiB for all configs and is dominated by weights and heads, so it says little about compute.
- (5) Turning has 9 windows from 4 scenes, so its CIs are wide.
