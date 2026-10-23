#!/usr/bin/env bash
# G1 Block-2 accelerators on Kaggle 2x T4 (sm75, fp16 autocast, SDPA). Run from the extracted tarball root.
#   bash run_accel_all.sh smoke   # 1 window per cell (~20-25 min incl. weight loads); check logs_smoke_*.txt, then:
#   bash run_accel_all.sh full    # all 70 windows, 11 cells (resumes: existing npz are skipped)
# Outputs: g1_accel_<mode>_outputs.tgz (run_meta + logs) and preds/accel_<mode>/<cell>/*.npz (send back for scoring).
set -euo pipefail
MODE=${1:-smoke}; cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PWD/vggt_src:${PYTHONPATH:-}" HF_HUB_DISABLE_TELEMETRY=1
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
pip install -q einops safetensors huggingface_hub 2>/dev/null || true   # NO torch pins (FastVGGT pin stripped)
python -c "from huggingface_hub import snapshot_download as s; s('facebook/VGGT-1B', allow_patterns=['*.json','model.safetensors'])"  # non-commercial, research only
python -m pytest -q tests/test_accel.py
MW=0; [ "$MODE" = smoke ] && MW=1
O=preds/accel_$MODE; mkdir -p $O
cell() { local g=$1 m=$2 b=$3 n=$4; local name=${m}_b${b}_N${n}
  CUDA_VISIBLE_DEVICES=$g python scripts/run_accel.py --method $m --budget $b --frames $n --max_windows $MW --out $O/$name 2>&1 | tee logs_${MODE}_${name}.txt; }
NG=$(python -c "import torch;print(torch.cuda.device_count())")
L0="fastvggt:0.5:32 gotohunt:0.5:32 come:0.25:16 fastvggt:0.1:8 gotohunt:0.1:8 avggt:0.25:16"
L1="come:0.5:32 gotohunt_all:0.5:32 fastvggt:0.25:16 gotohunt:0.25:16 come:0.1:8"
lane() { local g=$1; shift; for c in "$@"; do IFS=: read m b n <<< "$c"; cell $g $m $b $n || echo "CELL FAILED: $c" | tee -a logs_${MODE}_FAILED.txt; done; }
if [ "$NG" -ge 2 ]; then lane 0 $L0 & lane 1 $L1 & wait; else lane 0 $L0 $L1; fi
tar czf g1_accel_${MODE}_outputs.tgz $O/*/run_meta.json logs_${MODE}_*.txt manifests/accel_plan.json
echo "DONE $MODE -> g1_accel_${MODE}_outputs.tgz ; preds in $O"; ls logs_${MODE}_FAILED.txt 2>/dev/null && echo "SOME CELLS FAILED (see above)" || true
