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

## Step 2 — accelerators (main table + combo column)
Analytic FLOP fractions (results/block1/flop_table.csv; dense N32 = 113.4 TFLOP):

| Method | Native configs (frac) | Main 0.5 / 0.25 / 0.1 | Combo column (with frame drop) | sm80+? |
|---|---|---|---|---|
| AVGGT (our reimpl., t_early=9, σ∈{2,4,6,9}) | 0.644 / 0.564 / 0.536 / 0.515 | **not reachable (min = 0.515)** at all three | N16 σ=2..9: 0.278–0.247 (→0.25 cell: σ=9 N16 0.247); N8: 0.128–0.121 (0.1 not reachable) | No (pure PyTorch, explicit chunked attention; accuracy exact, speed not representative) |
| Global-token reduction family (FastVGGT; GoToHunt token part) | keep 0.46 → 0.5 | 0.5 reachable; 0.25/0.1 **not reachable (min = 0.324)** | N16 keep 0.596 → 0.25; N8 keep 0.404 → 0.1 | FastVGGT: no; GoToHunt: no custom kernel found (verify) |
| GoToHunt (official) | set via `--token_ds_layers`, `--global_as_frame_layers`, `--frame_token_downsample`; achieved FLOPs computed by our counter per config | as reached | as reached | verify on T4 |
| Co-Me (official) | merge ratio configs; kernels | as reached | as reached | **Effectively yes** (manual CUDA kernels; verified only on A100/Ada/5090) → accuracy on T4 via eager/unfused path if it exists, else Modal A100 |

AVGGT runs (T4, fp16):
```bash
for S in 2 4 9; do python scripts/run_vggt.py --method avggt --sigma $S --t_early 9 --out preds/avggt_s${S}_N32; done
python scripts/run_vggt.py --method avggt --sigma 9 --t_early 9 --stride 2 --out preds/avggt_s9_N16   # combo -> 0.247
python scripts/run_vggt.py --method avggt --sigma 9 --t_early 9 --stride 4 --out preds/avggt_s9_N8    # combo -> 0.121
```
FastVGGT / GoToHunt: clone official repos inside the notebook, wrap with `scripts/run_accel.py` (TO WRITE after a
15-min smoke test per repo) writing the same npz schema; GoToHunt frame selection (`--frame_strategy`) DISABLED so
the input frames equal the config's frames. Configure each to the budget via our FLOP counter; record achieved frac.

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
