# Block 1 GPU run plan (DRAFT — nothing sent/run on GPU yet)

Hardware: Kaggle 2x T4 16 GB (sm75). fp16 only (no bf16); torch SDPA (mem-efficient/math kernels on sm75, no
FlashAttention-2); `--attn xformers` fallback. Weights facebook/VGGT-1B: NON-COMMERCIAL, research use only.
Speed-ups not measured on sm80+ are reported as **calculated (analytic FLOPs)**, never as measured.

## Decisions applied (G1 Idea Agent, 2026-10-22)
1. Main table: method alone at budgets 0.5/0.25/0.1; unreachable cells = "not reachable (min = x)". A separate,
   labelled **combo column** (accelerator + frame drop) fills 0.25/0.1. Stop rule judged on the main table only.
2. AVGGT = **our reimplementation** (g1/avggt.py), FLOPs calculated.
3. GitHub repo: NOT created (needs Sudesh's own approval — see report).
4. >80 m: reported (coverage, n) but not scored. Far-field test = 40–80 m band. Primary CIs = scene-level bootstrap.

## Step 1 — controls (self-contained tarball)
`/workspace/g1_kaggle_controls.tar` -> upload as PRIVATE Kaggle dataset `g1-kaggle-controls`. In a 2xT4 notebook (internet ON for the HF weight download):
```bash
cp -r /kaggle/input/g1-kaggle-controls/g1_kaggle_controls /kaggle/working/ 2>/dev/null || tar xf /kaggle/input/g1-kaggle-controls/g1_kaggle_controls.tar -C /kaggle/working
cd /kaggle/working/g1_kaggle_controls && bash run_controls.sh sdpa      # fallback: bash run_controls.sh xformers
```
Runs dense_N32, dense_N16, dense_N8, dense_res0.5, dense_res0.25 over 70 windows (GPU0/GPU1 in parallel), a
10-window timing repeat of N32, then scores. Disk: weights 4.7 GB + bundle 0.4 GB + preds ~1.5 GB < 20 GB.
**Return to us:** `g1_controls_outputs.tgz` (results/block1/{per_window,summary,delta_vs_dense}.csv,
preds/*/run_meta.json with GPU name/torch/CUDA/per-window secs+peak mem, logs_*.txt) and, if space allows,
`preds/` npz (~1.5 GB) for re-scoring.

## Step 2 — accelerators (`g1_kaggle_accel.tar`, `run_accel_all.sh smoke` then `full`)
FLOP model replaced (g1/flops_accel.py): FastVGGT, GoToHunt and AVGGT shrink ONLY the global attention matmuls (qkv/proj/MLP
stay full), so K/V-only methods floor at ~0.46 at N32; Co-Me also merges before MLP. Plan (manifests/accel_plan.json):

| cell | knob (faithful) | achieved frac | budget met? |
|---|---|---|---|
| fastvggt 0.5/N32, 0.25/N16, 0.1/N8 | merging=0, merge_ratio=0.9 (paper; saturates: dst+10% protected never merge) | 0.580 / 0.263 / 0.124 | no / no / no (run at min) |
| gotohunt 0.5/N32, 0.25/N16, 0.1/N8 | paper/README: t=3, K/V ds layers 2-8, global-as-frame {0,1}, frame selection OFF | 0.816 / 0.319 / 0.137 | no / no / no (run at paper config) |
| gotohunt_all 0.5/N32 | official standalone path: ds in ALL non-gaf layers, t=4 (budget-forced) | 0.497 | yes |
| come 0.5/N32, 0.25/N16, 0.1/N8 | direct 3x3, enc layer 14, ranking threshold 0.447 / 0.292 / 0.352, input 294x504 | 0.495 / 0.247 / 0.099 | yes |
| avggt 0.25/N16 | sigma 9, t_early 9 (our reimpl.) | 0.247 | yes |

Co-Me on T4: pure-PyTorch port of the official direct mode (numerically exact at threshold 0 vs stock VGGT); the sm80+
fused kernels are NOT used, so T4 wall-clock for Co-Me is not representative — speed claim only from the Modal A100 check.

## Step 3 — Modal A100 check (Co-Me only, < 20 min)
Goal: one sm80 data point — does Co-Me's fused kernel build and what is its measured vs analytic speed-up at N=32.
```bash
# modal run (A100-40GB, image nvidia/cuda:12.8-devel + torch cu128), budget 20 min hard timeout
git clone https://github.com/co-me-tokens/CoMe && cd CoMe && <build kernels per README>   # ~8 min
python -m src.infer model=co_me_fused_vggt_3x3 dataset=custom_image <3 windows, N=32>          # ~5 min
python -m src.infer model=vggt dataset=custom_image <same 3 windows>                          # ~3 min
```
Record: build success, median latency both models, peak mem, and the predictions for those 3 windows (to score
accuracy with our scorer). Nothing else runs on Modal.

## GPU-hour estimate (T4)
| Item | GPU-h |
|---|---|
| Controls (5 configs ≈ 1.67 dense-equivalents × 70 windows × ~10 s) + timing repeat + weight download | ~1 |
| AVGGT 5 runs (explicit attention ≈ 1.5× dense time) | ~2.5 |
| FastVGGT + GoToHunt: ~5 configs each incl. combos | ~3.5 |
| Co-Me on T4 (eager path, if available) ~4 configs | ~2 |
| Env setup / kernel-build attempts / debug | ~4 |
| Margin ×1.3 | ~4 |
| **Total** | **≈ 15–17 GPU-h** (cap 30) |

### Accelerators (T4 estimate, 2026-10-23)
Per-window ≈ dense s/window × FLOP frac (+30% for Co-Me per-segment loop / FastVGGT similarity): ≈ 55 s per window over the
11 cells × 70 windows ≈ 1.1 h, + weight loads/smoke (11 cells × 1 window + loads) ≈ 0.4 h → **≈ 1.5 T4-h (budget 3 h with
margin)**; on 2x T4 two lanes ≈ 50 min wall. Cumulative G1 ≈ 3.1 + ≤3 = ≤6.1 of the 30 h/week cap.
