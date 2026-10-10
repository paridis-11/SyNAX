#!/usr/bin/env python3
"""Post-hoc SNN evaluation for FedNeurX CIFAR-10 and ImageNet checkpoints.

CNN checkpoints use Burst-ANN2SNN. ViT checkpoints use the repository's
native ViT-compatible ANN2SNN path. Training loops remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np

try:
    import medmnist  # noqa: F401
except ModuleNotFoundError:
    medmnist_stub = types.ModuleType("medmnist")
    medmnist_stub.INFO = {}
    sys.modules["medmnist"] = medmnist_stub

from Data import Data
from models.Model import ResidualBlock
from models.builder import build_model
from utils.utils import ann2snn
from utils_ann2snn import evaluate_snn, reset_net


def add_burst_paths(burst_root: str) -> None:
    burst_root = os.path.abspath(burst_root)
    for path in (
        os.path.join(burst_root, "neuromorphic"),
        os.path.join(burst_root, "static_datasets", "Cifar"),
        burst_root,
    ):
        if path not in sys.path:
            sys.path.insert(0, path)


def normalize_output(output):
    if isinstance(output, (list, tuple)):
        output = output[-1] if isinstance(output, list) else output[0]
    if output.dim() > 2:
        output = output.flatten(1)
    return output


class TensorOutputWrapper(nn.Module):
    """Make project models return a tensor for burst SpikeModel."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_output(self.model(x))


def default_args(saved: Dict, cli_args) -> SimpleNamespace:
    args = dict(saved or {})
    args.update(
        {
            "dataset": cli_args.dataset,
            "model": cli_args.model,
            "device": cli_args.device,
            "batchsize": cli_args.batch_size,
            "test_T": cli_args.test_T,
            "seed": cli_args.seed,
            "partition": cli_args.partition,
            "dir": cli_args.dir_alpha,
            "node_num": cli_args.node_num,
            "sample_ratio": cli_args.sample_ratio,
            "snn_ratio": cli_args.snn_ratio,
        }
    )
    args["num_classes"] = 1000 if cli_args.dataset == "imagenet" else 10
    args["classes"] = args["num_classes"]
    is_vit = "vit" in cli_args.model.lower()
    args["shape"] = 224 if is_vit or cli_args.dataset == "imagenet" else 32
    args["type"] = "VIT" if is_vit else "CNN"
    args.setdefault("all_data", True)
    args.setdefault("val_ratio", 0.1)
    args.setdefault("drop", 0.0)
    args.setdefault("R", 1)
    args.setdefault("E", 1)
    args.setdefault("lr", 1e-3)
    args.setdefault("model_config", "vgg16_cifar10.yaml")
    args.setdefault("drop_path_rate", 0.0)
    args.setdefault("opt_eps", 1e-8)
    args.setdefault("dyrep_recal_bn_iters", 20)
    args.setdefault("opt_no_filter", True)
    args.setdefault("optimizer", "adamw")
    args.setdefault("momentum", 0.0)
    args.setdefault("wandb", 0)
    args.setdefault("zero_momentum", 1.0)
    args.setdefault("linear_num", 8)
    args.setdefault("qkv_num", 8)
    args.setdefault("softmax_num", 8)
    args.setdefault("softmax_p", 0.0125)
    args.setdefault("zero_min", -0.5)
    args.setdefault("zero_max", 0.5)
    args.setdefault("zo_epsilon", 0.01)
    return SimpleNamespace(**args)


def extract_state_dict(ckpt: Dict) -> Dict[str, torch.Tensor]:
    for key in ("ann_model", "model", "state_dict"):
        if key in ckpt:
            state = ckpt[key]
            break
    else:
        state = ckpt
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


@torch.no_grad()
def evaluate_ann(model: nn.Module, loader, device: torch.device, max_batches: int = 0) -> float:
    model.to(device).eval()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        data = data.float().to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True)
        if target.dim() > 1:
            target = target.squeeze()
        pred = normalize_output(model(data)).argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def convert_to_native_snn_fp32(ann_model: nn.Module, loader, run_args) -> nn.Module:
    """Collect native ViT thresholds in FP32 to avoid AMP overflow."""
    from models.builder_snn import build_snn_model
    from utils.trans_utils import (
        replace_at_by_neuron,
        replace_nonlinear_by_neuron,
        replace_testneuron_by_twosideneuron,
    )
    from utils_ann2snn import replace_test_by_testneuron

    snn = build_snn_model(run_args).to(run_args.device)
    snn.load_state_dict(ann_model.state_dict(), strict=False)
    replace_test_by_testneuron(snn, 0.99)
    snn.eval()

    for data, _ in loader:
        data = data.float().to(run_args.device, non_blocking=True)
        snn(data)

    bad_thresholds = []
    threshold_count = 0
    for name, module in snn.named_modules():
        if module.__class__.__name__.lower() != "testneuron":
            continue
        threshold_count += 1
        for attr in ("scale_p", "scale_n"):
            value = getattr(module, attr)
            if not torch.isfinite(value).all():
                bad_thresholds.append(f"{name}.{attr}")
    print(
        f"native_fp32_thresholds={threshold_count} "
        f"nonfinite={len(bad_thresholds)}",
        flush=True,
    )
    if bad_thresholds:
        raise RuntimeError(
            "Non-finite native FP32 thresholds: " + ", ".join(bad_thresholds[:8])
        )

    replace_nonlinear_by_neuron(snn)
    replace_at_by_neuron(snn)
    replace_testneuron_by_twosideneuron(snn, run_args)
    return snn


@torch.no_grad()
def evaluate_native_snn_fp32(loader, model: nn.Module, device: torch.device, T: int):
    """Evaluate the native ViT SNN without autocast."""
    model.to(device).eval()
    correct = np.zeros(T, dtype=np.float64)
    total = 0
    for data, target in loader:
        data = data.float().to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True)
        if target.dim() > 1:
            target = target.squeeze()
        output = model(data, T=T)
        if len(output) != T:
            raise RuntimeError(f"Expected {T} native SNN outputs, got {len(output)}")
        for step, logits in enumerate(output):
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite native FP32 logits at step {step + 1}")
            correct[step] += logits.argmax(dim=1).eq(target).sum().item()
        total += target.numel()
        reset_net(model)
    return 100.0 * correct / max(total, 1)


class SpikeProjectResidualBlock(nn.Module):
    """Burst wrapper for this repo's models.Model.ResidualBlock."""

    def __init__(self, block: ResidualBlock, **spike_params):
        super().__init__()
        from general_utils.spiking_layer import SpikeModule
        from static_datasets.Cifar.models.resnet import SpikeResModule

        self.place1 = block.place1
        self.place2 = block.place2
        self.conv1 = SpikeModule(conv=block.conv1, **spike_params)
        self.conv1.add_module("relu", block.relu1)
        self.conv2 = SpikeResModule(conv=block.conv2, **spike_params)
        self.conv2.add_module("relu", block.relu)
        self.shortcut = block.shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = sum(self.place1(x))
        out = self.conv1(out)
        out = sum(self.place2(out))
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv2(out, residual)
        return out


def burst_specials(model_name: str):
    if model_name.lower() == "resnet18":
        return {ResidualBlock: SpikeProjectResidualBlock}
    return {}


def convert_to_burst_snn(ann_model: nn.Module, calib_loader, run_args, cli_args) -> nn.Module:
    from general_utils.calibration import bias_corr_model
    from general_utils.fold_bn import search_fold_and_remove_bn
    from general_utils.spiking_layer import SpikeModel, SpikeModule
    from general_utils.spiking_layer import get_maximum_activation

    model = TensorOutputWrapper(copy.deepcopy(ann_model)).to(run_args.device)
    search_fold_and_remove_bn(model)
    snn = SpikeModel(
        model=model,
        sim_length=cli_args.test_T,
        maxspike=cli_args.maxspike,
        specials=burst_specials(run_args.model),
    ).to(run_args.device)
    get_maximum_activation(
        calib_loader,
        model=snn,
        momentum=0.9,
        iters=cli_args.calib_iters,
        mse=(cli_args.calib != "none"),
        percentile=None,
        maxspike=cli_args.maxspike,
        sim_length=cli_args.test_T,
        channel_wise=True,
    )
    if cli_args.calib == "light":
        bias_corr_model(model=snn, train_loader=calib_loader, correct_mempot=False)
    for module in snn.modules():
        if isinstance(module, SpikeModule):
            module.spike_counter = 0
    snn.set_spike_state(use_spike=True)
    return snn


@torch.no_grad()
def evaluate_burst_snn(model: nn.Module, loader, device: torch.device, max_batches: int = 0) -> Tuple[float, float]:
    from general_utils.spiking_layer import SpikeModule
    try:
        from static_datasets.Cifar.models.resnet import SpikeResModule
        spike_types = (SpikeModule, SpikeResModule)
    except Exception:
        spike_types = (SpikeModule,)

    model.to(device).eval()
    correct = 0
    total = 0
    spikecount = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        data = data.float().to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True)
        if target.dim() > 1:
            target = target.squeeze()
        output = normalize_output(model(data))
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.numel()
        for module in model.modules():
            if isinstance(module, spike_types):
                spikecount += float(getattr(module, "spike_counter", 0.0))
                module.spike_counter = 0
    return 100.0 * correct / max(total, 1), spikecount / max(total, 1)


class LimitedLoader:
    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = max_batches

    def __iter__(self):
        for idx, batch in enumerate(self.loader):
            if idx >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.max_batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["cifar10", "imagenet"], default="cifar10")
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "vgg", "vgg16", "vit_small", "vit_base"],
    )
    parser.add_argument("--burst-root", default="/root/code/burst-ann2snn-master")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--node-num", type=int, default=50)
    parser.add_argument("--sample-ratio", type=float, default=0.2)
    parser.add_argument("--snn-ratio", type=float, default=0.5)
    parser.add_argument("--partition", choices=["iid", "dir"], default="iid")
    parser.add_argument("--dir-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-T", type=int, default=8)
    parser.add_argument("--maxspike", type=int, default=5)
    parser.add_argument("--calib-iters", type=int, default=16)
    parser.add_argument("--calib-batches", type=int, default=16)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--calib", choices=["none", "mse", "light"], default="mse")
    parser.add_argument(
        "--converter",
        choices=["auto", "burst", "native"],
        default="auto",
        help="auto selects Burst for CNNs and the native project converter for ViTs",
    )
    parser.add_argument(
        "--native-fp32",
        action="store_true",
        help="collect native ViT thresholds and evaluate the SNN in FP32",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    add_burst_paths(args.burst_root)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    run_args = default_args(ckpt.get("args", {}), args)
    run_args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = Data(run_args)
    ann = build_model(run_args, run_args.model).to(run_args.device)
    state = extract_state_dict(ckpt)
    missing, unexpected = ann.load_state_dict(state, strict=False)
    print(f"checkpoint={args.checkpoint}")
    print(f"model={run_args.model} partition={run_args.partition} node_num={run_args.node_num} sample_ratio={run_args.sample_ratio}")
    print(f"loaded: missing={len(missing)} unexpected={len(unexpected)}")

    ann_acc = evaluate_ann(ann, data.test_all, run_args.device, args.max_eval_batches)
    print(f"ANN eval: {ann_acc:.2f}%", flush=True)

    converter = args.converter
    if converter == "auto":
        converter = "native" if "vit" in args.model.lower() else "burst"
    if "vit" in args.model.lower() and converter == "burst":
        raise ValueError("Burst-ANN2SNN is not supported for ViT in this evaluator")
    print(f"converter={converter}", flush=True)

    if converter == "native":
        calibration_loader = data.test_all
        if args.dataset == "imagenet":
            calibration_loader = data.train_loader[0]
            if args.calib_batches:
                calibration_loader = LimitedLoader(calibration_loader, args.calib_batches)
        if args.native_fp32:
            print("native_precision=fp32", flush=True)
            native_snn = convert_to_native_snn_fp32(
                ann,
                calibration_loader,
                run_args,
            ).to(run_args.device)
            native_curve = evaluate_native_snn_fp32(
                data.test_all,
                native_snn,
                run_args.device,
                args.test_T,
            )
        else:
            print("native_precision=amp", flush=True)
            native_snn = ann2snn(ann, calibration_loader, run_args).to(run_args.device)
            native_curve = evaluate_snn(
                data.test_all,
                native_snn,
                run_args.device,
                args.test_T,
                run_args,
            )
        print(
            f"Native ANN2SNN eval: {float(native_curve[-1]):.2f}% | "
            f"curve={list(map(float, native_curve))}",
            flush=True,
        )
        return

    calib_dataset = data.train_loader[0].dataset
    calib_generator = torch.Generator()
    calib_generator.manual_seed(args.seed)
    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
        generator=calib_generator,
    )
    if args.calib_batches:
        calib_loader = LimitedLoader(calib_loader, args.calib_batches)
    burst_snn = convert_to_burst_snn(ann, calib_loader, run_args, args)
    burst_acc, avg_spikes = evaluate_burst_snn(burst_snn, data.test_all, run_args.device, args.max_eval_batches)
    print(f"Burst ANN2SNN eval: {burst_acc:.2f}% | avg_spike_counter_per_sample={avg_spikes:.2f}", flush=True)


if __name__ == "__main__":
    main()
