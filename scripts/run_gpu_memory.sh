#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export PYTHONPATH="$ROOT/static_cifar10${PYTHONPATH:+:$PYTHONPATH}"

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=(resnet18 vit_small vit_base vit_large)
fi

cd "$ROOT/static_cifar10"
"$PYTHON" benchmark_gpu_memory.py \
    --models "${MODELS[@]}" --device cuda:0 \
    --optimizer sgd --lr 0.05 --momentum 0.9 --weight-decay 0.0005 \
    --batch-size 4 --time-steps 8 --repeats 3 --seed 42 \
    --output "${OUTPUT:-$ROOT/results/memory/gpu_memory_benchmark_sgd_fp32_b4_T8.csv}"
