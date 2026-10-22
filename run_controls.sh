#!/usr/bin/env bash
# G1 Block-1 controls on Kaggle 2x T4 (sm75, fp16, SDPA). Self-contained: run from the extracted tarball root.
# Usage: bash run_controls.sh [ATTN=sdpa|xformers]   Outputs: preds/<cfg>/*.npz + run_meta.json, results/block1/*.csv
set -euo pipefail
ATTN=${1:-sdpa}
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/vggt_src:${PYTHONPATH:-}" HF_HUB_DISABLE_TELEMETRY=1
python - <<'PY'
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
pip install -q einops safetensors huggingface_hub 2>/dev/null || true
python -c "from huggingface_hub import snapshot_download as s; s('facebook/VGGT-1B', allow_patterns=['*.json','model.safetensors'])"  # ~4.7GB, non-commercial, research only
NG=$(python -c "import torch;print(torch.cuda.device_count())")
run() { CUDA_VISIBLE_DEVICES=$1 python scripts/run_vggt.py --attn "$ATTN" --dtype fp16 --out "preds/$2" "${@:3}" 2>&1 | tee "logs_$2.txt"; }
mkdir -p preds
if [ "$NG" -ge 2 ]; then
  ( run 0 dense_N32 --stride 1 ; run 0 dense_res0.25 --scale 0.25 ) &
  ( run 1 dense_N16 --stride 2 ; run 1 dense_N8 --stride 4 ; run 1 dense_res0.5 --scale 0.5 ) &
  wait
else
  run 0 dense_N32 --stride 1; run 0 dense_N16 --stride 2; run 0 dense_N8 --stride 4
  run 0 dense_res0.5 --scale 0.5; run 0 dense_res0.25 --scale 0.25
fi
# timing repeat of the reference on an otherwise idle GPU (wall-clock is T4-only)
run 0 dense_N32_timingrep --stride 1 --max_windows 10
python scripts/score.py preds/dense_N32 preds/dense_N16 preds/dense_N8 preds/dense_res0.5 preds/dense_res0.25 --out results/block1
tar czf g1_controls_outputs.tgz results/block1 preds/*/run_meta.json logs_*.txt
echo "DONE -> g1_controls_outputs.tgz (+ preds/ npz if requested)"
