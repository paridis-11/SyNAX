#!/usr/bin/env python3
"""Architecture-matched ResNet-18 event-learning baselines.

This runner reuses the SyNAX event cache, augmentation, IID client split, and
FedAvg protocol. S-TLLR and OTTT use their authors' ResNet-18 implementations.
The layer-wise STDP and ETLP-style options are diagnostic stage-local variants;
DECOLLE is adapted to ResNet-18 with fixed local classifiers and stop-gradient
boundaries between residual stages.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import synax_neuro as neuro
from train_central_burst_resnet_event import mix_batch


METHODS = (
    "stllr_bp",
    "stllr_local",
    "ottt_a",
    "ottt_o",
    "layerwise_stdp",
    "decolle_resnet18",
    "etlp_style_resnet18",
)


class EventConsistentAugment(Dataset):
    """Apply one spatial/temporal transform consistently to all event bins."""

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        sample = sample.float()
        padding = self.args.crop_padding
        if padding > 0:
            sample = F.pad(sample, (padding, padding, padding, padding))
            top = random.randint(0, 2 * padding)
            left = random.randint(0, 2 * padding)
            size = self.args.input_size
            sample = sample[:, top : top + size, left : left + size]
        if random.random() < self.args.hflip_prob:
            sample = torch.flip(sample, dims=(-1,))

        event = sample.reshape(2, 9, sample.shape[-2], sample.shape[-1])
        if self.args.temporal_shift > 0:
            shift = random.randint(-self.args.temporal_shift, self.args.temporal_shift)
            if shift:
                shifted = torch.zeros_like(event)
                if shift > 0:
                    shifted[:, shift:] = event[:, :-shift]
                else:
                    shifted[:, :shift] = event[:, -shift:]
                event = shifted
        if self.args.bin_drop_prob > 0:
            keep = torch.rand(9) >= self.args.bin_drop_prob
            if not keep.any():
                keep[random.randrange(9)] = True
            event = event * keep.view(1, 9, 1, 1)
        if self.args.polarity_drop_prob > 0 and random.random() < self.args.polarity_drop_prob:
            event[random.randrange(2)] = 0
        return event.reshape_as(sample), target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", required=True, choices=("cifar10dvs", "ncaltech101"))
    parser.add_argument("--data-path", default="/data/L202500420/FedNeurX/data")
    parser.add_argument("--stllr-root", default="/data/L202500420/FedNeurX/third_party/S-TLLR")
    parser.add_argument("--ottt-root", default="/data/L202500420/FedNeurX/third_party/OTTT-SNN")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-num", type=int, default=6)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--partition", choices=("iid", "dir"), default="iid")
    parser.add_argument("--dir-alpha", type=float, default=0.5)
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--E", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--sequence-normalization",
        choices=("none", "sample_max", "cifar10dvs_official", "channel_standardize"),
        default="sample_max",
    )
    parser.add_argument("--channel-mean", nargs=2, type=float, default=None)
    parser.add_argument("--channel-std", nargs=2, type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--crop-padding", type=int, default=8)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--temporal-shift", type=int, default=2)
    parser.add_argument("--bin-drop-prob", type=float, default=0.1)
    parser.add_argument("--polarity-drop-prob", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--delay-ls", type=int, default=5)
    parser.add_argument("--stllr-factors", nargs=4, type=float, default=None)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--ottt-small-input-stem", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ottt-loss-lambda", type=float, default=0.001)
    parser.add_argument("--persist-client-optimizer", type=int, choices=(0, 1), default=1)
    parser.add_argument("--diagnostics", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--record-spikes",
        type=int,
        choices=(0, 1),
        default=0,
        help="Count test-time Conv2dSTLLR output spikes at each evaluation.",
    )
    parser.add_argument("--local-loss-weight", type=float, default=1.0)
    parser.add_argument("--etlp-update-prob", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_data(args: argparse.Namespace):
    args.frame_data = 1
    args.speck_input = 0
    args.input_channels = 18
    args.input_scale = 1.0
    args.snn_ratio = 1.0
    data = neuro.build_neuro_data(args)
    if args.augment:
        data.train_loader = [
            DataLoader(
                EventConsistentAugment(loader.dataset, args),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=args.workers > 0,
            )
            for loader in data.train_loader
        ]
    return data


def voxel_to_sequence(voxel: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Recover [B,T,2,H,W] from SyNAX's [B,18,H,W] event cache."""
    steps = args.T
    batch, channels, height, width = voxel.shape
    if channels != 18:
        raise ValueError(f"Expected 18 event channels, got {channels}")
    sequence = voxel.view(batch, 2, 9, height, width).permute(0, 2, 1, 3, 4)
    if steps != 9:
        packed = sequence.permute(0, 2, 3, 4, 1).reshape(batch, 2 * height * width, 9)
        packed = F.interpolate(packed, size=steps, mode="linear", align_corners=False)
        sequence = packed.reshape(batch, 2, height, width, steps).permute(0, 4, 1, 2, 3)
    if args.sequence_normalization == "sample_max":
        scale = sequence.amax(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1.0)
        sequence = sequence / scale
    elif args.sequence_normalization == "cifar10dvs_official":
        if args.dataset != "cifar10dvs":
            raise ValueError("cifar10dvs_official normalization is only valid for CIFAR10-DVS")
        mean = sequence.new_tensor((0.2728, 0.1295)).view(1, 1, 2, 1, 1)
        std = sequence.new_tensor((0.2225, 0.1290)).view(1, 1, 2, 1, 1)
        sequence = (sequence - mean) / std
    elif args.sequence_normalization == "channel_standardize":
        if args.channel_mean is None or args.channel_std is None:
            raise ValueError("channel_standardize requires --channel-mean and --channel-std")
        mean = sequence.new_tensor(args.channel_mean).view(1, 1, 2, 1, 1)
        std = sequence.new_tensor(args.channel_std).view(1, 1, 2, 1, 1).clamp_min(1e-6)
        sequence = (sequence - mean) / std
    return sequence


class STLLRWrapper(nn.Module):
    def __init__(self, base: nn.Module, num_classes: int, local: bool, layerwise: bool, input_size: int):
        super().__init__()
        self.base = base
        self.num_classes = num_classes
        self.local = local
        self.layerwise = layerwise
        if local:
            from models.layers.feedback_alignment import TrainingHook

            stage_sizes = [input_size // 2]
            for _ in range(3):
                stage_sizes.append((stage_sizes[-1] + 1) // 2)
            dimensions = (
                (num_classes, 64, stage_sizes[0], stage_sizes[0]),
                (num_classes, 128, stage_sizes[1], stage_sizes[1]),
                (num_classes, 256, stage_sizes[2], stage_sizes[2]),
                (num_classes, 512, stage_sizes[3], stage_sizes[3]),
            )
            self.hooks = nn.ModuleList(
                [TrainingHook(num_classes, dim, "LocalLoss") for dim in dimensions]
            )
        else:
            self.hooks = nn.ModuleList()

        if layerwise:
            generator = torch.Generator().manual_seed(31991)
            heads = []
            for channels in (64, 128, 256, 512):
                head = nn.Linear(channels, num_classes, bias=False)
                nn.init.normal_(head.weight, generator=generator)
                head.weight.data.div_(head.weight.data.norm(dim=1, keepdim=True).clamp_min(1e-6))
                head.weight.requires_grad_(False)
                heads.append(head)
            self.local_heads = nn.ModuleList(heads)
        else:
            self.local_heads = nn.ModuleList()

    def reset_states(self) -> None:
        self.base.reset_states()

    def begin_spike_count(self) -> None:
        """Count binary outputs from every spiking convolution during evaluation."""
        device = next(self.parameters()).device
        self._spike_total = torch.zeros((), dtype=torch.float64, device=device)
        self._spike_handles = []

        def accumulate(_module, _inputs, output):
            self._spike_total = self._spike_total + output.detach().sum(dtype=torch.float64)

        for module in self.base.modules():
            if module.__class__.__name__ == "Conv2dSTLLR":
                self._spike_handles.append(module.register_forward_hook(accumulate))

    def end_spike_count(self) -> float:
        total = float(self._spike_total.item())
        for handle in self._spike_handles:
            handle.remove()
        del self._spike_handles, self._spike_total
        return total

    def forward(self, x: torch.Tensor, one_hot: Optional[torch.Tensor] = None):
        x = self.base.features.init_block(x)
        auxiliary = []
        for index in range(4):
            stage = getattr(self.base.features, f"stage{index + 1}")
            x = stage(x)
            if self.local and self.training and one_hot is not None:
                x = self.hooks[index](x, one_hot, None)
            if self.layerwise:
                pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
                auxiliary.append(self.local_heads[index](pooled))
                if index < 3:
                    x = x.detach()
        logits = self.base.output(self.base.final_pool(x).flatten(1))
        return logits, auxiliary


class StageLocalResNetWrapper(nn.Module):
    """ResNet-18 with fixed local readouts and stage-wise gradient boundaries."""

    def __init__(self, base: nn.Module, num_classes: int):
        super().__init__()
        self.base = base
        self.num_classes = num_classes
        generator = torch.Generator().manual_seed(27183)
        heads = []
        for channels in (64, 128, 256, 512):
            head = nn.Linear(channels, num_classes, bias=False)
            nn.init.normal_(head.weight, generator=generator)
            head.weight.data.div_(head.weight.data.norm(dim=1, keepdim=True).clamp_min(1e-6))
            head.weight.requires_grad_(False)
            heads.append(head)
        self.local_heads = nn.ModuleList(heads)

    def reset_states(self) -> None:
        self.base.reset_states()

    def forward(self, x: torch.Tensor):
        x = self.base.features.init_block(x)
        auxiliary = []
        for index in range(4):
            stage = getattr(self.base.features, f"stage{index + 1}")
            x = stage(x)
            pooled = F.adaptive_avg_pool2d(x, 1).flatten(1)
            auxiliary.append(self.local_heads[index](pooled))
            if index < 3:
                x = x.detach()
        logits = self.base.output(self.base.final_pool(x).flatten(1))
        return logits, auxiliary


def build_stllr(args: argparse.Namespace, num_classes: int) -> nn.Module:
    sys.path.insert(0, args.stllr_root)
    from models.ResNet_STDP import ResNet18

    factors = args.stllr_factors
    if factors is None:
        factors = [0.2, 0.5, -1.0, 1.0] if args.dataset in ("cifar10dvs", "ncaltech101") else [0.2, 0.75, -1.0, 1.0]
    base = ResNet18(
        in_channels=2,
        labels=num_classes,
        activation=None,
        factors=factors,
    )
    return STLLRWrapper(
        base,
        num_classes,
        local=args.method == "stllr_local",
        layerwise=args.method == "layerwise_stdp",
        input_size=args.input_size,
    )


def build_stage_local(args: argparse.Namespace, num_classes: int) -> nn.Module:
    sys.path.insert(0, args.stllr_root)
    if args.method == "decolle_resnet18":
        from models.ResNet_BPTT import ResNet18
    else:
        from models.ResNet_STDP import ResNet18

    base = ResNet18(
        in_channels=2,
        labels=num_classes,
        activation=None,
        factors=[0.2, 0.75, -1.0, 1.0],
    )
    return StageLocalResNetWrapper(base, num_classes)


def build_ottt(args: argparse.Namespace, num_classes: int) -> nn.Module:
    sys.path.insert(0, args.ottt_root)
    from models.spiking_resnet_imagenet import ScaledWSConv2d, online_spiking_nfresnet18
    from modules import neuron, surrogate

    model = online_spiking_nfresnet18(
        single_step_neuron=neuron.OnlineLIFNode,
        surrogate_function=surrogate.Sigmoid(alpha=4.0),
        c_in=2,
        num_classes=num_classes,
        grad_with_rate=True,
        tau=args.tau,
        neuron_dropout=args.drop_rate,
        drop_rate=args.drop_rate,
        v_reset=None,
        track_rate=True,
    )
    if args.ottt_small_input_stem:
        model.conv1 = ScaledWSConv2d(
            2, 64, kernel_size=3, stride=1, padding=1, bias=True, gain=True
        )
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(model.conv1.bias)
        model.maxpool = nn.Identity()
    return model


def build_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.method.startswith("ottt"):
        return build_ottt(args, num_classes)
    if args.method in ("decolle_resnet18", "etlp_style_resnet18"):
        return build_stage_local(args, num_classes)
    return build_stllr(args, num_classes)


def make_optimizer(model: nn.Module, args: argparse.Namespace, lr: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if args.optimizer == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=args.weight_decay)


def scheduled_lr(args: argparse.Namespace, round_index: int) -> float:
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * round_index / max(1, args.R)))


def mixed_ce(logits, targets_a, targets_b, lam, smoothing):
    return lam * F.cross_entropy(logits, targets_a, label_smoothing=smoothing) + (
        1.0 - lam
    ) * F.cross_entropy(logits, targets_b, label_smoothing=smoothing)


def ottt_loss(logits, targets_a, targets_b, lam, args):
    ce = mixed_ce(logits, targets_a, targets_b, lam, args.label_smoothing)
    if args.ottt_loss_lambda <= 0:
        return ce
    target = lam * F.one_hot(targets_a, logits.shape[1]).to(logits.dtype)
    target += (1.0 - lam) * F.one_hot(targets_b, logits.shape[1]).to(logits.dtype)
    mse = F.mse_loss(logits, target)
    return (1.0 - args.ottt_loss_lambda) * ce + args.ottt_loss_lambda * mse


def maybe_clip_grad(model: nn.Module, max_norm: float) -> float:
    if max_norm > 0:
        norm = float(
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm, error_if_nonfinite=True
            )
        )
    else:
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        norm = float(torch.stack(norms).norm(2)) if norms else 0.0
    if not math.isfinite(norm):
        raise FloatingPointError(f"Non-finite gradient norm: {norm}")
    return norm


def stllr_batch(model, sequence, targets_a, targets_b, lam, args, optimizer, training: bool):
    model.reset_states()
    one_hot = lam * F.one_hot(targets_a, num_classes=model.num_classes).float()
    one_hot += (1.0 - lam) * F.one_hot(targets_b, num_classes=model.num_classes).float()
    prediction = torch.zeros(targets_a.shape[0], model.num_classes, device=targets_a.device)
    losses = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    start = max(0, args.T - args.delay_ls)
    for timestep in range(args.T):
        logits, auxiliary = model(sequence[:, timestep], one_hot if training else None)
        prediction.add_(logits.detach())
        if training and timestep >= start:
            loss = mixed_ce(logits, targets_a, targets_b, lam, args.label_smoothing)
            if args.method == "layerwise_stdp" and auxiliary:
                local = torch.stack(
                    [mixed_ce(item, targets_a, targets_b, lam, args.label_smoothing) for item in auxiliary]
                ).mean()
                loss = loss + args.local_loss_weight * local
            loss = loss / max(1, args.T - start)
            loss.backward()
            losses.append(float(loss.detach().item()))
    if training:
        maybe_clip_grad(model, args.grad_clip)
        optimizer.step()
    return prediction, sum(losses)


def ottt_batch(model, sequence, targets_a, targets_b, lam, args, optimizer, training: bool):
    prediction = torch.zeros(targets_a.shape[0], model.fc.out_features, device=targets_a.device)
    losses = []
    online = args.method == "ottt_o"
    if training and not online:
        optimizer.zero_grad(set_to_none=True)
    for timestep in range(args.T):
        if training and online:
            optimizer.zero_grad(set_to_none=True)
        logits = model(sequence[:, timestep], init=timestep == 0)
        prediction.add_(logits.detach())
        if training:
            loss = ottt_loss(logits, targets_a, targets_b, lam, args) / args.T
            loss.backward()
            if online:
                maybe_clip_grad(model, args.grad_clip)
                optimizer.step()
            losses.append(float(loss.detach().item()))
    if training and not online:
        maybe_clip_grad(model, args.grad_clip)
        optimizer.step()
    return prediction, sum(losses)


def stage_local_batch(model, sequence, targets_a, targets_b, lam, args, optimizer, training: bool):
    model.reset_states()
    prediction = torch.zeros(targets_a.shape[0], model.num_classes, device=targets_a.device)
    losses = []
    if training:
        optimizer.zero_grad(set_to_none=True)

    for timestep in range(args.T):
        logits, auxiliary = model(sequence[:, timestep])
        prediction.add_(logits.detach())
        if not training:
            continue
        if args.method == "etlp_style_resnet18" and random.random() >= args.etlp_update_prob:
            continue
        local_losses = [
            mixed_ce(item, targets_a, targets_b, lam, args.label_smoothing)
            for item in auxiliary
        ]
        local_losses.append(mixed_ce(logits, targets_a, targets_b, lam, args.label_smoothing))
        loss = args.local_loss_weight * torch.stack(local_losses).mean() / args.T
        losses.append(loss)

    if training and losses:
        total_loss = torch.stack(losses).sum()
        total_loss.backward()
        maybe_clip_grad(model, args.grad_clip)
        optimizer.step()
        return prediction, float(total_loss.detach().item())
    return prediction, 0.0


def run_loader(model, loader, args, optimizer=None) -> Tuple[float, float, Optional[float]]:
    training = optimizer is not None
    model.train(training)
    total = correct = 0
    loss_sum = 0.0
    record_spikes = bool(not training and args.record_spikes and isinstance(model, STLLRWrapper))
    if record_spikes:
        model.begin_spike_count()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (voxel, targets) in enumerate(loader):
            limit = args.max_train_batches if training else args.max_eval_batches
            if limit and batch_index >= limit:
                break
            voxel = voxel.float().to(args.device, non_blocking=True)
            targets = targets.long().to(args.device, non_blocking=True)
            if training:
                voxel, targets_a, targets_b, lam = mix_batch(voxel, targets, args)
            else:
                targets_a, targets_b, lam = targets, targets, 1.0
            sequence = voxel_to_sequence(voxel, args)
            diagnostic_parameters = None
            if training and args.diagnostics and not getattr(args, "_diagnostic_emitted", False):
                trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
                selected = trainable[:1] + trainable[-1:]
                diagnostic_parameters = [
                    (name, parameter, parameter.detach().clone()) for name, parameter in selected
                ]
            if args.method.startswith("ottt"):
                prediction, loss = ottt_batch(
                    model, sequence, targets_a, targets_b, lam, args, optimizer, training
                )
            elif args.method in ("decolle_resnet18", "etlp_style_resnet18"):
                prediction, loss = stage_local_batch(
                    model, sequence, targets_a, targets_b, lam, args, optimizer, training
                )
            else:
                prediction, loss = stllr_batch(
                    model, sequence, targets_a, targets_b, lam, args, optimizer, training
                )
            if diagnostic_parameters is not None:
                histogram = torch.bincount(prediction.argmax(1), minlength=prediction.shape[1])
                details = {
                    "sequence_mean": float(sequence.mean()),
                    "sequence_std": float(sequence.std()),
                    "sequence_min": float(sequence.min()),
                    "sequence_max": float(sequence.max()),
                    "prediction_mean": float(prediction.mean()),
                    "prediction_std": float(prediction.std()),
                    "predicted_class_histogram": histogram.cpu().tolist(),
                    "loss": float(loss),
                    "parameters": {
                        name: {
                            "update_norm": float((parameter.detach() - before).norm()),
                            "gradient_norm": float(parameter.grad.detach().norm()) if parameter.grad is not None else None,
                        }
                        for name, parameter, before in diagnostic_parameters
                    },
                }
                print("DIAGNOSTIC " + json.dumps(details), flush=True)
                args._diagnostic_emitted = True
            correct += int(prediction.argmax(1).eq(targets).sum().item())
            total += targets.numel()
            loss_sum += loss * targets.numel()
    spikes_per_sample = model.end_spike_count() / max(total, 1) if record_spikes else None
    return loss_sum / max(total, 1), 100.0 * correct / max(total, 1), spikes_per_sample


def cpu_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def cpu_optimizer_state(optimizer) -> Dict:
    state = copy.deepcopy(optimizer.state_dict())
    for values in state["state"].values():
        for key, value in values.items():
            if torch.is_tensor(value):
                values[key] = value.detach().cpu()
    return state


def average_states(states: List[Dict[str, torch.Tensor]], reference: Dict[str, torch.Tensor]):
    result = copy.deepcopy(reference)
    for key, value in reference.items():
        if value.is_floating_point():
            result[key] = torch.stack([state[key].to(value.dtype) for state in states]).mean(0)
    return result


def main() -> None:
    args = parse_args()
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    data = configure_data(args)
    model = build_model(args, data.num_classes).to(args.device)
    output_dir = Path(args.save_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["device"] = str(config["device"])
    config.update(
        architecture="ResNet-18",
        input_protocol="same SyNAX cache; 18-channel voxel restored to T event frames",
        method_scope=(
            "stage-local diagnostic, not a canonical published ResNet STDP implementation"
            if args.method == "layerwise_stdp"
            else "DECOLLE adapted to ResNet-18 with fixed local readouts and stage stop-gradients"
            if args.method == "decolle_resnet18"
            else "ETLP-style stage-local approximation; not an official ETLP ResNet-18 implementation"
            if args.method == "etlp_style_resnet18"
            else "published method rule adapted to ResNet-18 FedAvg"
        ),
        parameters=sum(parameter.numel() for parameter in model.parameters()),
    )
    print("CONFIG " + json.dumps(config), flush=True)

    history = []
    best = -1.0
    optimizer_states: Dict[int, Dict] = {}
    sample_size = max(1, int(args.node_num * args.sample_ratio))
    for round_index in range(args.R):
        lr = scheduled_lr(args, round_index)
        sampled = random.sample(range(args.node_num), sample_size)
        local_states = []
        local_accs = []
        print(f"Round {round_index + 1}/{args.R}: sampled={sampled} lr={lr:.8g}", flush=True)
        for node_index in sampled:
            local = copy.deepcopy(model).to(args.device)
            optimizer = make_optimizer(local, args, lr)
            if args.persist_client_optimizer and node_index in optimizer_states:
                optimizer.load_state_dict(optimizer_states[node_index])
                for group in optimizer.param_groups:
                    group["lr"] = lr
            _, local_acc, _ = run_loader(local, data.train_loader[node_index], args, optimizer)
            if args.persist_client_optimizer:
                optimizer_states[node_index] = cpu_optimizer_state(optimizer)
            local_states.append(cpu_state(local))
            local_accs.append(local_acc)
            del optimizer, local
            torch.cuda.empty_cache()
        model.load_state_dict(average_states(local_states, cpu_state(model)), strict=True)

        if (round_index + 1) % args.eval_every == 0 or round_index + 1 == args.R:
            _, accuracy, spikes_per_sample = run_loader(model, data.test_loader, args)
            best = max(best, accuracy)
            row = {
                "round": round_index + 1,
                "snn_acc": accuracy,
                "best_snn": best,
                "mean_local_acc": float(np.mean(local_accs)),
            }
            if spikes_per_sample is not None:
                row["spikes_per_sample"] = spikes_per_sample
                row["spike_unit"] = "Conv2dSTLLR output spikes"
            history.append(row)
            print("EVAL " + json.dumps(row), flush=True)
            (output_dir / "history.json").write_text(json.dumps(history, indent=2))
            if accuracy >= best:
                torch.save(
                    {"round": round_index + 1, "best_acc": best, "model": cpu_state(model), "args": config},
                    output_dir / "best.pt",
                )
    print(f"Finished {args.method}: best_snn={best:.4f}%", flush=True)


if __name__ == "__main__":
    main()
