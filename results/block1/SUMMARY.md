# Block 1 (CPU prep) summary — 2026-10-22
- Windows: 70 non-overlapping 32-frame CAM_FRONT 12 Hz sweep windows from 10 nuScenes-mini scenes (0 skipped; 7/scene max).
  States (data/NOTES.md thresholds): straight 30, stationary 18, turning 9, other 13 (slow or mild curve; not forced into a class).
  Only 10 scenes -> report scene-cluster bootstrap CIs alongside window CIs; turning has n=9 from 4 scenes (weak).
- Eval subset: idx 0,4,...,28 (8 frames/window) — present in N, N/2, N/4. 550/560 eval frames have LiDAR within 30 ms (median 14.6 ms), ego-motion compensated.
- LiDAR pixel share by band: 0-15 m 57.1 %, 15-40 m 33.6 %, 40-80 m 9.0 %, >80 m 0.3 % (~9 px/frame) -> the >80 m band is essentially unmeasurable with nuScenes 32-beam LiDAR.
- Tests: 19/19 pass (test_results.txt). Overlay: overlay_scene-0061_w0_f4.png.
- FLOPs: flop_table.csv. Dense N32 = 113.4 TFLOP; N16 0.364, N8 0.148, res0.707 0.361, res0.5 0.133, res0.354 0.061. Global-only token reduction floor = 0.324.
- No GPU results yet. See ../../GPU_PLAN.md.
