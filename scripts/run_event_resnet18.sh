#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
METHOD=${1:?Usage: bash scripts/run_event_resnet18.sh METHOD DATASET [SEED]}
DATASET=${2:?DATASET must be cifar10dvs or ncaltech101}
SEED=${3:-42}
case "$DATASET" in cifar10dvs|ncaltech101) ;; *) exit 2 ;; esac
PY=${PYTHON_BIN:-python}
DATA=${DATA_ROOT:?Set DATA_ROOT to the directory containing the event frame caches}
BURST=${BURST_ROOT:-$ROOT/third_party/burst-ann2snn}
cd "$ROOT/event_resnet18"
mkdir -p logs checkpoints
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
TAG="${METHOD}_resnet18_${DATASET}_iid_R200_E1_T8_seed${SEED}"
COMMON=(--data-path "$DATA" --dataset "$DATASET" --input-size 128 --device cuda:0
  --seed "$SEED" --node-num 6 --R 200 --E 1 --sample-ratio 1.0 --partition iid
  --workers 4 --eval-every 5 --save-dir checkpoints --run-name "$TAG")
AUGMENT=(--label-smoothing 0.1 --grad-clip 1.0 --augment 1 --crop-padding 8
  --hflip-prob 0.5 --temporal-shift 2 --bin-drop-prob 0.1 --polarity-drop-prob 0.05
  --mixup-alpha 0.2 --cutmix-alpha 1.0 --mix-prob 1.0)
BURST_ARGS=(--burst-root "$BURST" --arch resnet18 --test-T 8 --calib light
  --calib-iters 5 --maxspike 4 --initialspike 4 --burst-search 0)
case "$METHOD" in
  synax)
    "$PY" -u synax_neuro.py "${COMMON[@]}" "${AUGMENT[@]}" "${BURST_ARGS[@]}" \
      --frame-data 1 --snn-ratio 0.67 --batch-size 8 --optimizer adamw --lr 0.001 \
      --weight-decay 5e-4 --warmup-rounds 0 --save-best 1 --zo-epsilon 0.01 \
      --zero-min -0.5 --zero-max 0.5 --zero-momentum 1 --eta-snn 0.005 \
      --feedback-batches 1 --snn-feedback-warmup-rounds 0 >"logs/$TAG.log" 2>&1 ;;
  burst_sgbptt)
    "$PY" -u bp_sg_burst_resnet_event.py "${COMMON[@]}" "${AUGMENT[@]}" "${BURST_ARGS[@]}" \
      --batch-size 32 --optimizer sgd --lr 0.05 --momentum 0.9 --weight-decay 5e-4 \
      --warmup-rounds 5 --surrogate-width 0.5 --surrogate-scale 1 --recalib-every 10 \
      >"logs/$TAG.log" 2>&1 ;;
  sew_sgbptt)
    "$PY" -u bp_sg_sew_resnet_event.py "${COMMON[@]}" "${AUGMENT[@]}" \
      --batch-size 8 --T 8 --optimizer sgd --lr 0.05 --momentum 0.9 \
      --weight-decay 5e-4 --warmup-rounds 0 --tau 2 --amp 1 >"logs/$TAG.log" 2>&1 ;;
  stllr_local)
    NORMALIZATION=sample_max
    if [[ "$DATASET" == cifar10dvs ]]; then NORMALIZATION=cifar10dvs_official; fi
    "$PY" -u resnet18_event_online_baselines.py "${COMMON[@]}" --method stllr_local \
      --stllr-root "$ROOT/event_resnet18/third_party/S-TLLR" --batch-size 8 --T 8 \
      --optimizer adam --lr 0.001 --weight-decay 0 --grad-clip 0 --label-smoothing 0 \
      --sequence-normalization "$NORMALIZATION" --augment 1 --crop-padding 8 \
      --hflip-prob 0.5 --temporal-shift 0 --bin-drop-prob 0 --polarity-drop-prob 0 \
      --mixup-alpha 0 --cutmix-alpha 0 --mix-prob 0 --persist-client-optimizer 1 \
      >"logs/$TAG.log" 2>&1 ;;
  *) echo 'METHOD must be synax, burst_sgbptt, sew_sgbptt, or stllr_local' >&2; exit 2 ;;
esac
echo "Completed: event_resnet18/logs/$TAG.log"
