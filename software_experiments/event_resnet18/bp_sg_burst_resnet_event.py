#!/usr/bin/env python3
"""Federated BP/SG on a Burst-ANN2SNN ResNet-18.

The forward path is the Burst-ANN2SNN multi-spike model. Only the derivative of
the hard floor/clamp spike count is replaced by a triangular surrogate during
backward. All clients train the converted SNN; no dense-ANN client is used.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import synax_neuro as burst
from train_central_burst_resnet_event import EventConsistentAugment, mix_batch


class BurstCountSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane_ratio, maxspike, width, scale):
        ctx.save_for_backward(membrane_ratio)
        ctx.maxspike = int(maxspike)
        ctx.width = float(width)
        ctx.scale = float(scale)
        return torch.floor(membrane_ratio).clamp(min=0, max=ctx.maxspike)

    @staticmethod
    def backward(ctx, grad_output):
        (membrane_ratio,) = ctx.saved_tensors
        width = max(ctx.width, 1e-6)
        derivative = torch.zeros_like(membrane_ratio)
        for level in range(1, ctx.maxspike + 1):
            distance = torch.abs(membrane_ratio - float(level)) / width
            derivative = derivative + torch.clamp(1.0 - distance, min=0.0)
        return grad_output * derivative * ctx.scale, None, None, None


def install_burst_surrogate() -> Tuple[type, type]:
    """Patch Burst modules while preserving their exact hard-spike forward."""
    from general_utils.spiking_layer import SpikeModule
    from general_utils.utils import StraightThrough
    from models.resnet import SpikeResModule

    def spike_module_forward(self, input_tensor):
        if self.analyze:
            x = self.fwd_func(input_tensor, self.weight, self.bias, **self.fwd_kwargs)
            return self.clip_floor(x, self.sim_length, self.threshold)
        if self.use_spike and not isinstance(self.relu, StraightThrough):
            self.cur_t += 1
            x = self.fwd_func(input_tensor, self.weight, self.bias, **self.fwd_kwargs)
            threshold = torch.as_tensor(self.threshold, device=x.device, dtype=x.dtype).detach().clamp_min(1e-6)
            x = x + 0.5 / self.sim_length * threshold
            self.mem_pot = self.mem_pot + x
            ratio = self.mem_pot / threshold
            count = BurstCountSurrogate.apply(
                ratio,
                int(self.maxspike),
                float(getattr(self, "surrogate_width", 0.5)),
                float(getattr(self, "surrogate_scale", 1.0)),
            )
            self.spike_counter += float(count.detach().sum().item())
            spike = count * threshold
            self.mem_pot = self.mem_pot - spike.detach()
            return spike

        # Layers without a following ReLU remain analog in the official Burst
        # model. Use their trainable weights while the SNN is being optimized.
        weight = self.weight if self.use_spike else self.org_weight
        bias = self.bias if self.use_spike else self.org_bias
        return self.relu(self.fwd_func(input_tensor, weight, bias, **self.fwd_kwargs))

    def spike_res_forward(self, input_tensor, residual):
        if self.analyze:
            x = self.fwd_func(input_tensor, self.weight, self.bias, **self.fwd_kwargs) + residual
            return self.clip_floor(x, self.sim_length, self.threshold)
        if self.use_spike:
            x = self.fwd_func(input_tensor, self.weight, self.bias, **self.fwd_kwargs) + residual
            threshold = torch.as_tensor(self.threshold, device=x.device, dtype=x.dtype).detach().clamp_min(1e-6)
            x = x + 0.5 / self.sim_length * threshold
            self.mem_pot = self.mem_pot + x
            ratio = self.mem_pot / threshold
            count = BurstCountSurrogate.apply(
                ratio,
                int(self.maxspike),
                float(getattr(self, "surrogate_width", 0.5)),
                float(getattr(self, "surrogate_scale", 1.0)),
            )
            self.spike_counter += float(count.detach().sum().item())
            self.cur_t += 1
            spike = count * threshold
            self.mem_pot = self.mem_pot - spike.detach()
            return spike
        return self.relu(self.fwd_func(input_tensor, self.org_weight, self.org_bias, **self.fwd_kwargs) + residual)

    SpikeModule.forward = spike_module_forward
    SpikeResModule.forward = spike_res_forward
    return SpikeModule, SpikeResModule


def parse_args():
    parser = argparse.ArgumentParser(description="Burst ResNet-18 surrogate-gradient FedAvg")
    parser.add_argument("--burst-root", default="/data/L202500420/burst-ann2snn-master")
    parser.add_argument("--data-path", default="/data/L202500420/FedNeurX/data")
    parser.add_argument("--dataset", choices=["cifar10dvs", "ncaltech101"], required=True)
    parser.add_argument("--arch", default="resnet18", choices=["resnet18"])
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-num", type=int, default=6)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--partition", choices=["iid", "dir"], default="iid")
    parser.add_argument("--dir-alpha", type=float, default=0.5)
    parser.add_argument("--R", type=int, default=100)
    parser.add_argument("--E", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--test-T", type=int, default=8)
    parser.add_argument("--maxspike", type=int, default=4)
    parser.add_argument("--initialspike", type=int, default=4)
    parser.add_argument("--calib", choices=["none", "light"], default="light")
    parser.add_argument("--calib-iters", type=int, default=5)
    parser.add_argument("--burst-search", type=int, default=0)
    parser.add_argument("--searchtime", type=int, default=1)
    parser.add_argument("--maxspike-ratio", type=float, default=1.0)
    parser.add_argument("--minspike", type=int, default=1)
    parser.add_argument("--search-method", default="pruning")
    parser.add_argument("--metric", default="kl")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--augment", type=int, default=0)
    parser.add_argument("--crop-padding", type=int, default=8)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--temporal-shift", type=int, default=2)
    parser.add_argument("--bin-drop-prob", type=float, default=0.1)
    parser.add_argument("--polarity-drop-prob", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--surrogate-width", type=float, default=0.5)
    parser.add_argument("--surrogate-scale", type=float, default=1.0)
    parser.add_argument("--recalib-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_data_args(args):
    args.frame_data = 1
    args.snn_ratio = 1.0
    args.speck_input = 0
    args.input_channels = 18
    args.input_scale = 1.0
    return args


def add_train_augmentation(data, args):
    if not args.augment:
        return
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


def configure_surrogate(model, spike_types, args):
    count = 0
    for module in model.modules():
        if isinstance(module, spike_types):
            module.surrogate_width = args.surrogate_width
            module.surrogate_scale = args.surrogate_scale
            module.spike_counter = 0.0
            count += 1
    model.set_spike_state(use_spike=True)
    return count


def sync_analog_copies(model, spike_types):
    for module in model.modules():
        if isinstance(module, spike_types):
            module.org_weight = module.weight.detach().clone()
            module.org_bias = None if module.bias is None else module.bias.detach().clone()


def recalibrate_thresholds(model, loader, spike_types, args):
    from general_utils.spiking_layer import get_maximum_activation

    sync_analog_copies(model, spike_types)
    get_maximum_activation(
        loader,
        model=model,
        momentum=0.9,
        iters=args.calib_iters,
        mse=True,
        percentile=None,
        maxspike=args.maxspike,
        sim_length=args.test_T,
        channel_wise=True,
    )
    configure_surrogate(model, spike_types, args)


def make_optimizer(model, args, lr):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)


def scheduled_lr(initial_lr, round_idx, rounds, warmup_rounds):
    if warmup_rounds > 0 and round_idx < warmup_rounds:
        return initial_lr * float(round_idx + 1) / warmup_rounds
    progress = (round_idx - warmup_rounds) / max(1, rounds - warmup_rounds)
    return initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def normalize_output(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    return output.flatten(1) if output.dim() > 2 else output


def train_local(model, loader, args, lr, print_grad_check=False):
    model.to(args.device).train()
    model.set_spike_state(use_spike=True)
    optimizer = make_optimizer(model, args, lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    total_loss = 0.0
    correct = 0
    total = 0
    grad_checked = False
    for _ in range(args.E):
        for batch_idx, (data, target) in enumerate(loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            data = data.float().to(args.device, non_blocking=True)
            target = target.long().to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            data, target_a, target_b, lam = mix_batch(data, target, args)
            output = normalize_output(model(data))
            loss = lam * criterion(output, target_a) + (1.0 - lam) * criterion(output, target_b)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if print_grad_check and not grad_checked:
                finite = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
                print(f"Surrogate gradient check: norm={float(grad_norm):.8e} finite={finite}", flush=True)
                if not finite or float(grad_norm) <= 0.0:
                    raise RuntimeError("Burst surrogate gradient is zero or non-finite")
                grad_checked = True
            optimizer.step()
            total_loss += float(loss.item()) * target.numel()
            correct += int(output.argmax(1).eq(target).sum().item())
            total += target.numel()
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, args, spike_types):
    model.to(args.device).eval()
    model.set_spike_state(use_spike=True)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    spikes = 0.0
    for batch_idx, (data, target) in enumerate(loader):
        if args.max_eval_batches and batch_idx >= args.max_eval_batches:
            break
        data = data.float().to(args.device, non_blocking=True)
        target = target.long().to(args.device, non_blocking=True)
        output = normalize_output(model(data))
        loss = criterion(output, target)
        total_loss += float(loss.item()) * target.numel()
        correct += int(output.argmax(1).eq(target).sum().item())
        total += target.numel()
        for module in model.modules():
            if isinstance(module, spike_types):
                spikes += float(module.spike_counter)
                module.spike_counter = 0.0
    return (
        total_loss / max(total, 1),
        100.0 * correct / max(total, 1),
        spikes / max(total, 1),
    )


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def average_states(states, reference):
    averaged = copy.deepcopy(reference)
    for key, value in reference.items():
        if not torch.is_floating_point(value):
            continue
        averaged[key] = torch.stack([state[key].to(dtype=value.dtype) for state in states]).mean(0)
    return averaged


def save_checkpoint(path, model, args, round_idx, best_acc, spike_types):
    runtime = {}
    for name, module in model.named_modules():
        if isinstance(module, spike_types):
            runtime[name] = {
                "threshold": torch.as_tensor(module.threshold).detach().cpu(),
                "maxspike": int(module.maxspike),
            }
    saved_args = vars(args).copy()
    saved_args["device"] = str(saved_args["device"])
    torch.save(
        {
            "round": round_idx,
            "best_acc": best_acc,
            "model": cpu_state(model),
            "burst_runtime": runtime,
            "args": saved_args,
        },
        path,
    )


def main():
    args = configure_data_args(parse_args())
    burst.add_burst_paths(args.burst_root)
    spike_module, spike_res_module = install_burst_surrogate()
    spike_types = (spike_module, spike_res_module)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print(
        f"Burst BP/SG: dataset={args.dataset} arch={args.arch} init=scratch "
        f"clients={args.node_num} participation={args.sample_ratio} T={args.test_T} maxspike={args.maxspike}",
        flush=True,
    )
    print(
        f"optimizer={args.optimizer} lr={args.lr} E={args.E} surrogate_width={args.surrogate_width} "
        f"surrogate_scale={args.surrogate_scale} recalib_every={args.recalib_every} "
        f"augment={bool(args.augment)} mixup={args.mixup_alpha} cutmix={args.cutmix_alpha}",
        flush=True,
    )
    data = burst.build_neuro_data(args)
    add_train_augmentation(data, args)
    ann = burst.build_neuro_model(args.arch, data.num_classes).to(args.device)
    snn = burst.convert_ann_to_burst_snn(ann, data.calib_loader, args)
    count = configure_surrogate(snn, spike_types, args)
    print(f"Built Burst ResNet SNN: spike_modules={count}", flush=True)
    del ann
    torch.cuda.empty_cache()

    checkpoint_dir = Path(args.save_dir) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    sample_size = max(1, int(args.node_num * args.sample_ratio))

    initial_loss, initial_acc, initial_spikes = evaluate(snn, data.test_loader, args, spike_types)
    print(
        f"Initial SNN Acc@1={initial_acc:.2f}% loss={initial_loss:.4f} "
        f"avg_burst_spikes={initial_spikes:.2f}",
        flush=True,
    )

    for round_idx in range(args.R):
        lr = scheduled_lr(args.lr, round_idx, args.R, args.warmup_rounds)
        sampled = random.sample(range(args.node_num), sample_size)
        reference = cpu_state(snn)
        local_states: List[Dict[str, torch.Tensor]] = []
        local_losses = []
        local_accs = []
        print(f"=============== Burst BP/SG Round {round_idx + 1}/{args.R}: sampled={sampled} lr={lr:.6g} ===============", flush=True)
        for local_position, node_idx in enumerate(sampled):
            local_model = copy.deepcopy(snn).to(args.device)
            loss, acc = train_local(
                local_model,
                data.train_loader[node_idx],
                args,
                lr,
                print_grad_check=(round_idx == 0 and local_position == 0),
            )
            local_states.append(cpu_state(local_model))
            local_losses.append(loss)
            local_accs.append(acc)
            del local_model
            torch.cuda.empty_cache()

        snn.load_state_dict(average_states(local_states, reference), strict=True)
        if args.recalib_every and (round_idx + 1) % args.recalib_every == 0:
            recalibrate_thresholds(snn, data.calib_loader, spike_types, args)

        if (round_idx + 1) % args.eval_every == 0 or (round_idx + 1) == args.R:
            loss, acc, avg_spikes = evaluate(snn, data.test_loader, args, spike_types)
            best_acc = max(best_acc, acc)
            print(
                f"Round {round_idx + 1} Burst BP/SG Acc@1={acc:.2f}% loss={loss:.4f} "
                f"avg_burst_spikes={avg_spikes:.2f} local_loss={np.mean(local_losses):.4f} "
                f"local_acc={np.mean(local_accs):.2f}% best={best_acc:.2f}%",
                flush=True,
            )
            if acc >= best_acc:
                save_checkpoint(
                    checkpoint_dir / "best_burst_bpsg.pt",
                    snn,
                    args,
                    round_idx + 1,
                    best_acc,
                    spike_types,
                )

    print(f"Finished Burst BP/SG. best_acc={best_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
