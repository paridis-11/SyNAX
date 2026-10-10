#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${1:?Usage: bash scripts/run_static_cifar10.sh MODEL [SEED]}
SEED=${2:-42}
case "$MODEL" in
  resnet18) ROUNDS=100; CONVERTER=burst ;;
  vit_small|vit_base) ROUNDS=500; CONVERTER=native ;;
  *) echo 'MODEL must be resnet18, vit_small, or vit_base' >&2; exit 2 ;;
esac
PY=${PYTHON_BIN:-python}
PARTITION=${PARTITION:-iid}
DIR_ALPHA=${DIR_ALPHA:-0.5}
NODES=${NODES:-50}
BURST=${BURST_ROOT:-$ROOT/third_party/burst-ann2snn}
TAG="synax_snn_cifar10_${MODEL}_${PARTITION}_n${NODES}_R${ROUNDS}_E5_T8_seed${SEED}_eta01"
cd "$ROOT/static_cifar10"
mkdir -p logs checkpoints
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export SYNAX_ETA_SNN=0.1 SYNAX_RUN_NAME="$TAG"
"$PY" -u SyNAX_normalized_snn.py --dataset cifar10 --model "$MODEL" \
  --node_num "$NODES" --sample_ratio 0.2 --snn_ratio 0.5 --R "$ROUNDS" --E 5 \
  --batchsize 32 --test_T 8 --zero_momentum 1 --wandb 0 --seed "$SEED" \
  --device cuda:0 --partition "$PARTITION" --dir "$DIR_ALPHA" \
  --optimizer adamw --lr 0.001 --weight-decay 0.001 --shape 224 \
  >"logs/train_${TAG}.log" 2>&1
"$PY" -u eval_cifar10_snn_checkpoint.py --checkpoint "checkpoints/$TAG/best_ann.pt" \
  --model "$MODEL" --device cuda:0 --batch-size 32 --node-num "$NODES" \
  --sample-ratio 0.2 --snn-ratio 0.5 --partition "$PARTITION" --dir-alpha "$DIR_ALPHA" \
  --seed "$SEED" --test-T 8 --maxspike 5 --calib-iters 16 --calib-batches 16 \
  --converter "$CONVERTER" --burst-root "$BURST" >"logs/eval_${TAG}.log" 2>&1
echo "Completed: static_cifar10/logs/train_${TAG}.log and eval_${TAG}.log"
