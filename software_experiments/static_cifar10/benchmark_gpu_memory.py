#!/usr/bin/env python3
"""Reproducible peak-memory benchmark for BP/SG and SyNAX executors.

The benchmark measures total process-local CUDA tensor memory with
``torch.cuda.max_memory_allocated``. It also records reserved memory for
diagnostics, but the paper figure should use ``peak_allocated_gib``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())

import models.builder_snn
import models.model_vit
from models.builder_snn import build_snn_model
from utils.trans_utils import AtNeuron, TwoSideNeuron
from utils.utils import ann2snn


class PatchedVisionTransformer(models.model_vit.VisionTransformer):
    """Accept timm-style names used by ``builder_snn``."""

    def __init__(self, **kwargs):
        if "drop_rate" in kwargs:
            kwargs["drop_ratio"] = kwargs.pop("drop_rate")
        if "attn_drop_rate" in kwargs:
            kwargs["attn_drop_ratio"] = kwargs.pop("attn_drop_rate")
        super().__init__(**kwargs)


models.builder_snn.VisionTransformer = PatchedVisionTransformer


def safe_at_neuron_forward(self, x, y):
    """Autograd-safe equivalent of the original AtNeuron recurrence."""
    if self.t == 0:
        self.tot_a = x
        self.tot_b = y
        self.tot_t = x @ y
        self.t = 1
        return x @ y

    self.tot_t = self.tot_t + (x @ y + x @ self.tot_b + self.tot_a @ y)
    self.tot_a = self.tot_a + x
    self.tot_b = self.tot_b + y
    self.t += 1
    return (
        (x @ self.tot_b + self.tot_a @ y - x @ y) / (self.t - 1)
        - self.tot_t / (self.t * (self.t - 1))
    )


class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return (value > 0).to(value.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        return grad_output * (value.abs() < 0.5).to(value.dtype)


class SurrogateNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem = None
        self.v_th = 0.5
        self.decay = 0.5

    def forward(self, value):
        if self.mem is None:
            self.mem = torch.zeros_like(value)
        self.mem = self.mem * self.decay + value
        spike = SurrogateHeaviside.apply(self.mem - self.v_th)
        self.mem = self.mem * (1 - spike)
        return [spike]

    def reset(self):
        self.mem = None


def replace_with_surrogate(module):
    for name, child in list(module.named_children()):
        if isinstance(child, TwoSideNeuron):
            setattr(module, name, SurrogateNeuron())
        else:
            replace_with_surrogate(child)
    return module


class DummyLoader:
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __iter__(self):
        yield self.data, self.target

    def __len__(self):
        return 1


class ModelArgs:
    def __init__(self, model_name, device, batch_size, time_steps):
        self.device = device
        self.dataset = "cifar10"
        self.model = model_name
        self.num_classes = 10
        self.linear_num = 2
        self.qkv_num = 2
        self.softmax_num = 2
        self.softmax_p = 0.1
        self.zero_momentum = 0.9
        self.seed = 42
        self.shape = 224 if "vit" in model_name.lower() else 32
        self.batchsize = batch_size
        self.test_T = time_steps
        self.device_ratio = "1:1:1"
        self.node_num = 10
        self.zo_epsilon = 0.01


@dataclass
class MemoryResult:
    model: str
    method: str
    optimizer: str
    repeat: int
    batch_size: int
    time_steps: int
    input_size: int
    parameter_mib: float
    baseline_allocated_gib: float
    peak_allocated_gib: float
    incremental_peak_gib: float
    peak_reserved_gib: float
    elapsed_seconds: float


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize(device):
    torch.cuda.synchronize(device)


def clear_cuda(device):
    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)


def reset_peak(device):
    clear_cuda(device)
    torch.cuda.reset_peak_memory_stats(device)


def gib(value_bytes):
    return value_bytes / (1024**3)


def memory_snapshot(device):
    return {
        "allocated": gib(torch.cuda.memory_allocated(device)),
        "peak_allocated": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": gib(torch.cuda.max_memory_reserved(device)),
    }


def parameter_mib(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)


def final_logits(output, time_steps):
    return output[-1] / time_steps if isinstance(output, list) else output


def reset_network_state(model):
    for module in model.modules():
        if module is model:
            continue
        reset = getattr(module, "reset", None)
        if callable(reset):
            try:
                reset()
            except TypeError:
                pass


def build_converted_snn(args, loader):
    base_model = build_snn_model(args).to(args.device)
    converted = ann2snn(base_model, loader, args)
    del base_model
    return converted.to(args.device)


def make_optimizer(name, model, cli_args):
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cli_args.lr,
            momentum=cli_args.momentum,
            weight_decay=cli_args.weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay
        )
    return torch.optim.Adam(
        model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay
    )


def measure_bp_sg(model_name, repeat, data, target, loader, args, cli_args):
    original_at_forward = AtNeuron.forward
    if "vit" in model_name.lower():
        AtNeuron.forward = safe_at_neuron_forward

    model = optimizer = loss = output = None
    try:
        model = replace_with_surrogate(build_converted_snn(args, loader))
        model.train()
        optimizer = make_optimizer(cli_args.optimizer, model, cli_args)
        criterion = nn.CrossEntropyLoss()

        reset_peak(args.device)
        baseline = gib(torch.cuda.memory_allocated(args.device))
        synchronize(args.device)
        started = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        output = model(data, T=args.test_T)
        loss = criterion(final_logits(output, args.test_T), target)
        loss.backward()
        optimizer.step()

        synchronize(args.device)
        elapsed = time.perf_counter() - started
        snapshot = memory_snapshot(args.device)
        return MemoryResult(
            model=model_name,
            method="BP/SG",
            optimizer=cli_args.optimizer,
            repeat=repeat,
            batch_size=args.batchsize,
            time_steps=args.test_T,
            input_size=args.shape,
            parameter_mib=parameter_mib(model),
            baseline_allocated_gib=baseline,
            peak_allocated_gib=snapshot["peak_allocated"],
            incremental_peak_gib=snapshot["peak_allocated"] - baseline,
            peak_reserved_gib=snapshot["peak_reserved"],
            elapsed_seconds=elapsed,
        )
    finally:
        AtNeuron.forward = original_at_forward
        del model, optimizer, loss, output
        clear_cuda(args.device)


def apply_direction(model, direction, scale):
    with torch.no_grad():
        for parameter, delta in zip(model.parameters(), direction):
            parameter.add_(delta, alpha=scale)


def measure_synax(model_name, repeat, data, target, loader, args, cli_args):
    model = output = loss = direction = None
    try:
        model = build_converted_snn(args, loader)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        direction = [torch.randn_like(parameter) for parameter in model.parameters()]

        reset_peak(args.device)
        baseline = gib(torch.cuda.memory_allocated(args.device))
        synchronize(args.device)
        started = time.perf_counter()

        with torch.no_grad():
            apply_direction(model, direction, cli_args.zo_epsilon)
            reset_network_state(model)
            output = model(data, T=args.test_T)
            loss_plus = criterion(final_logits(output, args.test_T), target).item()
            del output

            apply_direction(model, direction, -2 * cli_args.zo_epsilon)
            reset_network_state(model)
            output = model(data, T=args.test_T)
            loss_minus = criterion(final_logits(output, args.test_T), target).item()

            apply_direction(model, direction, cli_args.zo_epsilon)
            _directional_derivative = (loss_plus - loss_minus) / (
                2 * cli_args.zo_epsilon
            )

        synchronize(args.device)
        elapsed = time.perf_counter() - started
        snapshot = memory_snapshot(args.device)
        return MemoryResult(
            model=model_name,
            method="SyNAX",
            optimizer="none",
            repeat=repeat,
            batch_size=args.batchsize,
            time_steps=args.test_T,
            input_size=args.shape,
            parameter_mib=parameter_mib(model),
            baseline_allocated_gib=baseline,
            peak_allocated_gib=snapshot["peak_allocated"],
            incremental_peak_gib=snapshot["peak_allocated"] - baseline,
            peak_reserved_gib=snapshot["peak_reserved"],
            elapsed_seconds=elapsed,
        )
    finally:
        del model, output, loss, direction
        clear_cuda(args.device)


def mean(values: Iterable[float]):
    values = list(values)
    return sum(values) / len(values)


def sample_sd(values: Iterable[float]):
    values = list(values)
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return (sum((value - average) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_summary(path, rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row.model, row.method, row.optimizer), []).append(row)

    fields = [
        "model",
        "method",
        "optimizer",
        "n",
        "peak_allocated_gib_mean",
        "peak_allocated_gib_sd",
        "peak_reserved_gib_mean",
        "incremental_peak_gib_mean",
        "elapsed_seconds_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, group in grouped.items():
            peaks = [row.peak_allocated_gib for row in group]
            writer.writerow(
                {
                    "model": key[0],
                    "method": key[1],
                    "optimizer": key[2],
                    "n": len(group),
                    "peak_allocated_gib_mean": f"{mean(peaks):.6f}",
                    "peak_allocated_gib_sd": f"{sample_sd(peaks):.6f}",
                    "peak_reserved_gib_mean": f"{mean(row.peak_reserved_gib for row in group):.6f}",
                    "incremental_peak_gib_mean": f"{mean(row.incremental_peak_gib for row in group):.6f}",
                    "elapsed_seconds_mean": f"{mean(row.elapsed_seconds for row in group):.6f}",
                }
            )


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet18", "vit_small", "vit_base", "vit_large"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--zo-epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/gpu_memory_benchmark.csv"))
    return parser.parse_args()


def main():
    cli_args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device(cli_args.device)
    torch.cuda.set_device(device)
    torch.autograd.set_detect_anomaly(cli_args.detect_anomaly)
    rows = []

    for model_name in cli_args.models:
        for repeat in range(1, cli_args.repeats + 1):
            set_seed(cli_args.seed + repeat - 1)
            model_args = ModelArgs(
                model_name, device, cli_args.batch_size, cli_args.time_steps
            )
            data = torch.randn(
                model_args.batchsize,
                3,
                model_args.shape,
                model_args.shape,
                device=device,
            )
            target = torch.randint(
                0, model_args.num_classes, (model_args.batchsize,), device=device
            )
            loader = DummyLoader(data, target)

            for measure in (measure_bp_sg, measure_synax):
                result = measure(
                    model_name, repeat, data, target, loader, model_args, cli_args
                )
                rows.append(result)
                print(
                    "RESULT "
                    f"model={result.model} method={result.method} repeat={repeat} "
                    f"peak_allocated={result.peak_allocated_gib:.3f} GiB "
                    f"peak_reserved={result.peak_reserved_gib:.3f} GiB "
                    f"elapsed={result.elapsed_seconds:.3f}s",
                    flush=True,
                )

            del data, target, loader
            clear_cuda(device)

    raw_path = cli_args.output
    summary_path = raw_path.with_name(f"{raw_path.stem}_summary.csv")
    metadata_path = raw_path.with_name(f"{raw_path.stem}_metadata.json")
    write_csv(raw_path, rows)
    write_summary(summary_path, rows)
    metadata_path.write_text(
        json.dumps(
            {
                "command": " ".join(sys.argv),
                "git_commit": git_commit(),
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
                "device_capability": torch.cuda.get_device_capability(device),
                "detect_anomaly": cli_args.detect_anomaly,
                "measurement": "torch.cuda.max_memory_allocated",
                "unit": "GiB (bytes / 1024^3)",
                "arguments": vars(cli_args) | {"output": str(cli_args.output)},
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {raw_path}, {summary_path}, and {metadata_path}")


if __name__ == "__main__":
    main()
