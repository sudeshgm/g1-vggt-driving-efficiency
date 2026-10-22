# Block 1 GPU run plan (DRAFT — nothing has been sent or run on GPU)

Target: Kaggle 2x T4 16 GB (sm75): fp16 only (no bf16), no FlashAttention-2. Attention through torch SDPA
(mem-efficient / math backend on sm75); xformers memory_efficient_attention as fallback. VGGT-1B weights: facebook/VGGT-1B,
non-commercial licence, research use only. All speed-ups not measured on sm80+ are reported as ANALYTIC (FLOP counter).

## Data bundle (≈415 MB, well under 20 GB)
`/workspace/g1_block1_bundle.tar` = bundle/images (2240 CAM_FRONT sweep JPEGs), bundle/depth (550 sparse LiDAR depth npz,
eval frames only), bundle/manifests (windows.json, frames_stride{1,2,4}.txt). Upload as a private Kaggle dataset `g1-block1-bundle`.

## Session setup
```bash
cd /kaggle/working && tar xf /kaggle/input/g1-block1-bundle/g1_block1_bundle.tar
git clone https://github.com/facebookresearch/vggt && pip install -e vggt   # torch already present
# copy repo code (scripts/, g1/) from the private bundle / dataset; repo is NOT on GitHub yet
python -c "import torch;print(torch.cuda.get_device_capability())"   # expect (7, 5)
```

## Controls (stock VGGT-1B) — each GPU of the 2x T4 runs one config at a time
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_vggt.py --stride 1 --out preds/dense_N32 --img_root bundle/images
CUDA_VISIBLE_DEVICES=1 python scripts/run_vggt.py --stride 2 --out preds/dense_N16 --img_root bundle/images
CUDA_VISIBLE_DEVICES=1 python scripts/run_vggt.py --stride 4 --out preds/dense_N8  --img_root bundle/images
CUDA_VISIBLE_DEVICES=0 python scripts/run_vggt.py --scale 0.5   --out preds/dense_res0.5
CUDA_VISIBLE_DEVICES=0 python scripts/run_vggt.py --scale 0.25  --out preds/dense_res0.25    # 1/4 res: 70x126 px, 5x9 patches
CUDA_VISIBLE_DEVICES=0 python scripts/run_vggt.py --scale 0.707 --out preds/dense_res0.707   # optional, cheap
python scripts/score.py preds/* --out results/block1
```
Timing: run dense_N32 a second time with `--out preds/dense_N32_rep` for wall-clock variance; report T4 wall-clock as T4-only.

## Accelerators (released code). Budgets 0.5 / 0.25 / 0.1 of dense N=32 VGGT FLOPs (analytic, results/block1/flop_table.csv)
| Method | Code | sm80+ needed? | Plan on T4 |
|---|---|---|---|
| GoToHunt (2605.23892) | github.com/zsh2000/gotohunt (MapAnything env) | No custom kernel found in README — believed sm75-OK, verify | `--token_ds_layers`, `--global_as_frame_layers`, `--frame_token_downsample` swept to hit budget. Its frame-selection (`--frame_strategy diverse`, MegaLoc covisibility) must be DISABLED or fixed to the 32 frames, else it changes the eval set. |
| FastVGGT | github.com/mystorm16/FastVGGT | No (pure PyTorch merging; uses SDPA) — verify | `--merging <block>` + merge ratio swept. |
| AVGGT (2512.02541) | **repo not located yet** — blocker if none released | unknown | If no code: reimplement (global→frame in early layers + K/V subsample) as masking; FLOPs analytic. |
| Co-Me | github.com/co-me-tokens/CoMe | **Yes in practice**: custom CUDA kernels compiled manually, verified only on A100/Ada/5090 (CUDA 12.8/13.0); new "customized flash attn" path | Try compile on T4; if it fails use its non-fused/eager path (or reimplement merge with masking) for ACCURACY only; speed-up = analytic. Candidate for the one Modal A100 sm80 check (fused kernel latency, ~20 min). |

Wrapper pattern (to write once each repo is checked on GPU): `scripts/run_accel.py --method {gotohunt,fastvggt,avggt,come} --budget {0.5,0.25,0.1} --out preds/<method>_b<budget>`
must write the same npz schema as run_vggt.py (idx, extrinsic, depth, input_hw, secs, peak_mem) and log the achieved analytic FLOP fraction.

**Budget feasibility (important):** at N=32, 294x518 (777 patches/frame), global attention is 67.6 % of FLOPs; encoder+frame
blocks+heads are a 32.4 % floor. Methods that only reduce global-attention tokens CANNOT reach 0.25 or 0.1. AVGGT-style with 12
global layers turned to frame attention cannot go below ~0.47 by KV subsampling alone. Only methods that also cut frame-block/encoder
tokens (GoToHunt's frame-token downsampling, Co-Me merging across blocks) might. Unreachable cells will be reported as "not
reachable (min achievable = x)" — never silently replaced. Decision needed from idea agent: allow accelerator + frame-drop combos to fill 0.25/0.1?

## GPU-hour estimate (T4)
Dense N=32 = 113 TFLOP/window (analytic); assume ~10 s/window on T4 fp16 incl. I/O.
- Controls (6 configs, 2.1 dense-equivalents x 70 windows) + timing repeat: ~1 GPU-h
- Accelerators 4 x 3 budgets, masked/eager implementations often cost >= dense on T4: ~12 runs x 70 x ~15 s ≈ 3.5 GPU-h
- Env setup/kernel compile attempts/debug/model load (Co-Me, GoToHunt-MapAnything env): ~4 GPU-h
- Re-runs/failures margin x1.5
**Total ≈ 12–16 GPU-h** (within 15–25 target, under the 30 h/week cap). Spare ~10 h can go to the π3 column or random-merge seeds.
