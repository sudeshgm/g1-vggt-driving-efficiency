# G1 failure report: VGGT accelerators on forward-motion driving

**Verdict: KILL**, by the pre-registered rule. At matched or lower FLOPs, plain frame drop (feeding VGGT fewer frames) matches or beats every accelerator we tested. No accelerator damages far-field depth (40-80 m) more than near-field depth (0-15 m).

Frozen result: tag [`block1-final`](https://github.com/sudeshgm/g1-vggt-driving-efficiency/tree/block1-final) (commit `884412e`).
Source tables: [results/block1_final/block1_final_tables.md](block1_final/block1_final_tables.md), [final_main_table.csv](block1_final/final_main_table.csv), [final_paired_deltas.csv](block1_final/final_paired_deltas.csv), [final_far_vs_near.csv](block1_final/final_far_vs_near.csv), [summary.csv](block1_final/summary.csv), plot [acc_vs_flops_final.png](block1_final/acc_vs_flops_final.png). Controls: [results/block1/SUMMARY.md](block1/SUMMARY.md), [block1_tables.md](block1/block1_tables.md).

Every number below is copied from those files. Brackets are 95% scene-level bootstrap CIs.

## 1. Verdict in plain words

- The question was whether VGGT speed-up methods hold up on driving video, where the camera mostly moves forward. We compared them at equal compute against the simplest option: using fewer frames.
- Using fewer frames costs almost nothing. With 8 of 32 frames (0.148 of the FLOPs), every metric stays within the CI of the full 32-frame run.
- FastVGGT, Co-Me and our AVGGT reimplementation are worse than frame drop beyond the CI on at least one metric, usually on several.
- GoToHunt in its paper setting never reaches its FLOP budget. It ties frame drop on depth and the point map, with small "wins" that don't matter in practice: ATE is better by about 0.005 m, and in some cells AUC is better by about 0.001. The automatic Q1 check in block1_final_tables.md flags these as "YES". We count them as a tie, not a reason to continue, because the gains are far smaller than the cost of the extra complexity.
- GoToHunt forced onto all layers (to hit the 0.5 budget) collapses.
- Far vs near: the "far worse beyond CI?" column is **no** in all 11 accelerator cells.

## 2. Key numbers

FLOP fractions are calculated (analytic), relative to dense N32 at 294x518, which is **113.4 TFLOP**. **Global attention is 67.6%** of that. The decision set has 61 windows (turning excluded). AUC@30 also excludes stationary windows, leaving 43 windows.

### Frame-drop baselines and resolution controls (decision set)

| config | FLOP frac | AUC@30 | ATE (m) | PtMap rel | AbsRel 40-80 | T4 s/window |
|---|---|---|---|---|---|---|
| dense N32 | 1.000 | 0.952 [0.941, 0.963] | 0.098 [0.042, 0.168] | 0.201 [0.176, 0.226] | 0.239 [0.191, 0.287] | 12.35 |
| dense N16 | 0.364 | 0.952 [0.941, 0.962] | 0.101 [0.042, 0.173] | 0.202 [0.176, 0.227] | 0.239 [0.191, 0.287] | 4.64 |
| dense N8 | 0.148 | 0.951 [0.940, 0.961] | 0.100 [0.044, 0.169] | 0.202 [0.176, 0.227] | 0.240 [0.193, 0.288] | 2.06 |
| res0.5 (N32) | 0.133 | 0.638 [0.625, 0.649] | 0.194 [0.101, 0.303] | 0.391 [0.349, 0.434] | 0.366 [0.303, 0.435] | 1.96 |
| res0.25 (N32) | 0.030 | 0.471 [0.382, 0.541] | 2.140 [0.793, 3.907] | 0.716 [0.585, 0.834] | 0.601 [0.536, 0.672] | 0.61 |

AbsRel by depth band (all 70 windows, from block1_final/summary.csv, state=all):

| config | 0-15 m | 15-40 m | 40-80 m |
|---|---|---|---|
| dense N32 | 0.105 [0.093, 0.120] | 0.119 [0.105, 0.133] | 0.248 [0.196, 0.302] |
| dense N16 | 0.104 [0.092, 0.119] | 0.120 [0.106, 0.135] | 0.248 [0.196, 0.302] |
| dense N8 | 0.102 [0.092, 0.117] | 0.121 [0.106, 0.136] | 0.249 [0.198, 0.303] |
| res0.5 | 0.181 [0.165, 0.198] | 0.181 [0.150, 0.215] | 0.374 [0.310, 0.444] |

- **Frame drop:** N8 vs N32 paired deltas are all under 0.003 and every CI includes 0. For example, AbsRel 40-80 m is +0.001 [-0.002, +0.004] (block1/SUMMARY.md).
- **Resolution control:** res0.5 costs about the same as N8 (0.133 FLOPs) but is much worse. AUC@30 is −0.247 [0.168, 0.313] vs N32, AbsRel 40-80 m is +0.126 [+0.098, +0.159], and ATE roughly doubles. Lowering resolution is the wrong way to cut compute.

### Accelerators: FLOPs reached and paired deltas vs frame drop

Deltas are accelerator − baseline. A negative ΔAUC is worse. Positive ΔATE, ΔPtMap or ΔAbsRel is worse. "worse"/"better" means beyond the scene CI; "tie" means the CI includes 0.

| method | cell (budget, frames) | FLOP reached | met budget? | vs | Δ AUC@30 | Δ ATE (m) | Δ PtMap rel | Δ AbsRel 40-80 |
|---|---|---|---|---|---|---|---|---|
| FastVGGT | 0.5, N32 | 0.580 | missed | N8 (0.148) | −0.013 [−0.017, −0.009] worse | +0.014 [+0.003, +0.030] worse | +0.001 tie | +0.006 [−0.000, +0.012] tie |
| FastVGGT | 0.25, N16 | 0.262 | missed | N16 (0.364) | −0.011 [−0.015, −0.009] worse | +0.027 [+0.003, +0.064] worse | +0.003 tie | +0.009 [+0.003, +0.016] worse |
| FastVGGT | 0.1, N8 | 0.124 | missed | N8 (0.148) | −0.009 [−0.012, −0.006] worse | +0.043 [+0.016, +0.082] worse | +0.009 [+0.002, +0.016] worse | +0.016 [+0.007, +0.025] worse |
| GoToHunt (paper) | 0.5, N32 | 0.816 | missed | N8 (0.148) | +0.002 [+0.001, +0.002] better | −0.006 [−0.013, −0.001] better | −0.001 tie | −0.003 tie |
| GoToHunt (paper) | 0.25, N16 | 0.319 | missed | N16 (0.364) | +0.000 tie | −0.005 [−0.010, −0.001] better | −0.000 tie | −0.001 tie |
| GoToHunt (paper) | 0.1, N8 | 0.137 | missed | N8 (0.148) | +0.001 tie | −0.006 [−0.012, −0.001] better | −0.002 tie | −0.002 tie |
| GoToHunt (all layers) | 0.5, N32 | 0.497 | met | N8 (0.148) | −0.544 [−0.606, −0.481] worse | +3.076 [+1.581, +4.656] worse | +0.343 [+0.233, +0.452] worse | +0.036 [+0.017, +0.057] worse |
| Co-Me | 0.5, N32 | 0.495 | met | N8 (0.148) | −0.030 [−0.067, +0.002] tie | +0.109 [+0.044, +0.200] worse | +0.091 [+0.057, +0.128] worse | +0.079 [+0.040, +0.118] worse |
| Co-Me | 0.25, N16 | 0.247 | met | N16 (0.364) | −0.005 tie | +0.043 [+0.019, +0.079] worse | +0.026 [+0.006, +0.050] worse | +0.040 [+0.019, +0.063] worse |
| Co-Me | 0.1, N8 | 0.099 | met | N8 (0.148) | −0.017 tie | +0.071 [+0.026, +0.136] worse | +0.050 [+0.024, +0.079] worse | +0.058 [+0.033, +0.084] worse |
| AVGGT (our reimpl.) | 0.25, N16 | 0.247 | met | N16 (0.364) | −0.023 [−0.029, −0.018] worse | +0.033 [+0.014, +0.057] worse | +0.016 [+0.010, +0.023] worse | +0.021 [+0.016, +0.027] worse |

Notes:
- AVGGT was not reachable at the 0.5 budget at N32 (minimum 0.515) or at N8 (minimum 0.121), so it has a single cell.
- N8 always reaches lower FLOPs (0.148) than the N32 accelerator cells (0.495–0.816), yet it ties or beats them. For the full table, which also compares each cell against N16, see [block1_final_tables.md](block1_final/block1_final_tables.md).
- Far minus near relative AbsRel change: no cell is worse beyond the CI. Examples:
  - Co-Me 0.1/N8: +1.6% [−16.4%, +17.6%]
  - AVGGT: +2.2% [−4.1%, +8.6%]
  - GoToHunt all-layers: −65.0% [−127.4%, −10.9%], meaning its damage is mostly near-field.

**GPU use:** about 5.7 T4-hours of the 30 T4-hour budget (Kaggle 2x T4, fp16, SDPA).

## 3. Caveats

- **Small data.** Only 10 nuScenes-mini scenes: 70 windows of 32 CAM_FRONT frames at 12 Hz. The CIs are scene-level bootstraps over 10 clusters.
- **Scoring favours frame drop.** Only frames 0,4,…,28 are scored, the frames every config sees. These are exactly N8's inputs, so frame drop is never asked to recover a frame it skipped. A stride-2 check (N16 vs N32 on 0,2,…,30) also tied, but those frames are still N16's own inputs.
- **Turning windows are too few** (n=9 from 4 scenes). They are reported as weak and excluded from the decision. Stationary windows (n=18) get no AUC.
- **>80 m is unmeasurable.** It is 0.3% of LiDAR pixels (~9 px/frame) with the 32-beam LiDAR, so it is reported but not scored.
- **Co-Me** kernels need sm80+, and the T4 is sm75. Its T4 timing is therefore invalid, and its accuracy comes from our pure-PyTorch port of direct mode with the official predictor checkpoint. Co-Me's own eval masks merged regions; we score all pixels, as pre-registered.
- **AVGGT** is our own reimplementation, not the authors' code.
- **FLOPs are calculated, not measured.** Measured T4 speed-ups differ, e.g. N8 is 6.0x on the T4 vs 6.7x calculated.
- **Weights:** VGGT-1B is licensed non-commercial, for research use only.

## 4. What not to retry, and what might still be worth testing

Don't retry:
- Accelerators that only touch global attention, at N≤32 on driving video. Frame drop already removes most of the cost at no accuracy loss, and token reduction has a hard floor (0.324 of the FLOPs).
- Matched-FLOP grids at short sequence lengths. At N≤32, the frame-drop baseline is too strong to beat.
- Framing the work around the far field without denser long-range depth ground truth. nuScenes LiDAR gives 9% of pixels at 40-80 m and almost none beyond 80 m.

Might still be worth testing:
- Long sequences (≥200 frames). The quadratic global-attention cost dominates there, and dropping frames may start to lose real information.
- Scoring on the skipped frames, i.e. making frame drop recover frames it did not see, with interpolated poses and depth or a held-out frame protocol.
