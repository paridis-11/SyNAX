"""Federated surrogate-gradient training with the CIFAR10-DVS SEW-ResNet.

The network follows the official CIFAR10-DVS SEW-ResNet/Wide-7B design from
Fang et al. (NeurIPS 2021), adapted to the current SpikingJelly API and the
FedNeurX six-client experiment protocol.
"""

import argparse
import copy
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import bp_sg_burst_resnet_event as common
import synax_neuro as neuro


def parse_args():
    parser = argparse.ArgumentParser(description="SEW-ResNet surrogate-gradient FedAvg")
    parser.add_argument("--vendor-path", default="vendor/spikingjelly_pkg")
    parser.add_argument("--data-path", default="/data/L202500420/FedNeurX/data")
    parser.add_argument("--dataset", choices=["cifar10dvs", "ncaltech101"], required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-num", type=int, default=6)
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--partition", choices=["iid", "dir"], default="iid")
    parser.add_argument("--dir-alpha", type=float, default=0.5)
    parser.add_argument("--R", type=int, default=200)
    parser.add_argument("--E", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--T", type=int, default=8)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--amp", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def load_spikingjelly(vendor_path):
    path = os.path.abspath(vendor_path)
    if path not in sys.path:
        sys.path.insert(0, path)
    from spikingjelly.activation_based import functional, layer, neuron, surrogate

    return functional, layer, neuron, surrogate


def build_sew_resnet(num_classes, tau, sj):
    functional, layer, neuron, surrogate = sj

    def plif():
        return neuron.ParametricLIFNode(
            init_tau=tau,
            v_threshold=1.0,
            detach_reset=True,
            surrogate_function=surrogate.ATan(),
            step_mode="m",
            backend="torch",
        )

    def conv3x3(in_channels, out_channels):
        return nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            plif(),
        )

    def conv1x1(in_channels, out_channels):
        return nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ),
            plif(),
        )

    class SEWBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv = nn.Sequential(conv3x3(channels, channels), conv3x3(channels, channels))

        def forward(self, x):
            return self.conv(x) + x

    class SEWResNetEvent(nn.Module):
        def __init__(self):
            super().__init__()
            modules = []
            in_channels = 2
            # Official CIFAR10-DVS SEW-ResNet: seven residual stages with
            # temporal LIF neurons and spatial pooling after every stage.
            for channels in (64, 64, 64, 64, 128, 128, 128):
                if in_channels != channels:
                    modules.append(conv1x1(in_channels, channels))
                    in_channels = channels
                modules.append(SEWBlock(channels))
                modules.append(layer.SeqToANNContainer(nn.MaxPool2d(2, 2)))
            modules.append(nn.Flatten(2))
            self.features = nn.Sequential(*modules)
            self.head = nn.Linear(128, num_classes)

        def forward(self, frames):
            # Dataset batches are [N,T,2,H,W]; SpikingJelly uses [T,N,...].
            sequence = frames.permute(1, 0, 2, 3, 4).contiguous()
            features = self.features(sequence)
            return self.head(features.mean(0))

    model = SEWResNetEvent()
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    return model


def configure_data(args):
    args.frame_data = 1
    args.snn_ratio = 1.0
    args.speck_input = 0
    args.input_channels = 18
    args.input_scale = 1.0
    data = neuro.build_neuro_data(args)
    common.add_train_augmentation(data, args)
    return data


def to_temporal_frames(inputs, time_steps):
    """Convert the shared [N,18,H,W] cache into [N,T,2,H,W]."""
    n, channels, height, width = inputs.shape
    if channels != 18:
        raise ValueError(f"Expected 18 event channels, got {inputs.shape}")
    frames = inputs.reshape(n, 2, 9, height, width)
    if time_steps != 9:
        flat = frames.permute(0, 1, 3, 4, 2).reshape(-1, 1, 9)
        flat = F.interpolate(flat, size=time_steps, mode="linear", align_corners=False)
        flat = flat * (9.0 / float(time_steps))
        frames = flat.reshape(n, 2, height, width, time_steps).permute(0, 4, 1, 2, 3)
    else:
        frames = frames.permute(0, 2, 1, 3, 4)
    return frames.contiguous()


class SpikeMeter:
    def __init__(self, model, neuron_type):
        self.enabled = False
        self.total = 0.0
        self.handles = [module.register_forward_hook(self._hook) for module in model.modules() if isinstance(module, neuron_type)]

    def _hook(self, _module, _inputs, output):
        if self.enabled:
            self.total += float(output.detach().sum().item())

    def reset(self):
        self.total = 0.0

    def close(self):
        for handle in self.handles:
            handle.remove()


def make_optimizer(model, args, lr):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)


def train_local(model, loader, args, lr, functional, grad_check=False):
    model.to(args.device).train()
    optimizer = make_optimizer(model, args, lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp))
    total_loss = 0.0
    correct = 0
    total = 0
    checked = False
    for _ in range(args.E):
        for batch_idx, (inputs, targets) in enumerate(loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            inputs = inputs.float().to(args.device, non_blocking=True)
            targets = targets.long().to(args.device, non_blocking=True)
            inputs, target_a, target_b, lam = common.mix_batch(inputs, targets, args)
            frames = to_temporal_frames(inputs, args.T)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp)):
                output = model(frames)
                loss = lam * criterion(output, target_a) + (1.0 - lam) * criterion(output, target_b)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if grad_check and not checked:
                finite = bool(torch.isfinite(torch.as_tensor(grad_norm)).item())
                print(f"Surrogate gradient check: norm={float(grad_norm):.8e} finite={finite}", flush=True)
                if not finite or float(grad_norm) <= 0.0:
                    raise RuntimeError("SEW surrogate gradient is zero or non-finite")
                checked = True
            scaler.step(optimizer)
            scaler.update()
            functional.reset_net(model)
            total_loss += float(loss.item()) * targets.numel()
            correct += int(output.detach().argmax(1).eq(targets).sum().item())
            total += targets.numel()
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, args, functional, neuron_type):
    model.to(args.device).eval()
    criterion = nn.CrossEntropyLoss()
    meter = SpikeMeter(model, neuron_type)
    meter.reset()
    meter.enabled = True
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if args.max_eval_batches and batch_idx >= args.max_eval_batches:
            break
        inputs = inputs.float().to(args.device, non_blocking=True)
        targets = targets.long().to(args.device, non_blocking=True)
        frames = to_temporal_frames(inputs, args.T)
        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            output = model(frames)
            loss = criterion(output, targets)
        total_loss += float(loss.item()) * targets.numel()
        correct += int(output.argmax(1).eq(targets).sum().item())
        total += targets.numel()
        functional.reset_net(model)
    meter.enabled = False
    avg_spikes = meter.total / max(total, 1)
    meter.close()
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1), avg_spikes


def save_checkpoint(path, model, args, round_idx, best_acc):
    payload = {
        "round": round_idx,
        "best_acc": best_acc,
        "model": common.cpu_state(model),
        "args": vars(args).copy(),
    }
    payload["args"]["device"] = str(payload["args"]["device"])
    torch.save(payload, path)


def main():
    args = parse_args()
    sj = load_spikingjelly(args.vendor_path)
    functional, _layer, neuron, _surrogate = sj
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    common.set_seed(args.seed)
    data = configure_data(args)
    model = build_sew_resnet(data.num_classes, args.tau, sj).to(args.device)
    neuron_count = sum(isinstance(module, neuron.ParametricLIFNode) for module in model.modules())
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"SEW BP/SG: dataset={args.dataset} architecture=official-cifar10dvs-SEW-ResNet "
        f"clients={args.node_num} participation={args.sample_ratio} T={args.T} params={params}",
        flush=True,
    )
    print(
        f"optimizer={args.optimizer} lr={args.lr} E={args.E} batch={args.batch_size} "
        f"warmup={args.warmup_rounds} augment={bool(args.augment)} mixup={args.mixup_alpha} cutmix={args.cutmix_alpha}",
        flush=True,
    )
    print(f"Built SEW-ResNet: ParametricLIFNode={neuron_count} connect_f=ADD", flush=True)

    checkpoint_dir = Path(args.save_dir) / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    sample_size = max(1, int(args.node_num * args.sample_ratio))
    initial_loss, initial_acc, initial_spikes = evaluate(
        model, data.test_loader, args, functional, neuron.ParametricLIFNode
    )
    print(
        f"Initial SEW Acc@1={initial_acc:.2f}% loss={initial_loss:.4f} avg_lif_spikes={initial_spikes:.2f}",
        flush=True,
    )

    for round_idx in range(args.R):
        lr = common.scheduled_lr(args.lr, round_idx, args.R, args.warmup_rounds)
        sampled = random.sample(range(args.node_num), sample_size)
        reference = common.cpu_state(model)
        local_states: List[Dict[str, torch.Tensor]] = []
        local_losses = []
        local_accs = []
        print(f"=============== SEW BP/SG Round {round_idx + 1}/{args.R}: sampled={sampled} lr={lr:.6g} ===============", flush=True)
        for local_position, node_idx in enumerate(sampled):
            local_model = copy.deepcopy(model).to(args.device)
            loss, acc = train_local(
                local_model,
                data.train_loader[node_idx],
                args,
                lr,
                functional,
                grad_check=(round_idx == 0 and local_position == 0),
            )
            local_states.append(common.cpu_state(local_model))
            local_losses.append(loss)
            local_accs.append(acc)
            del local_model
            torch.cuda.empty_cache()
        model.load_state_dict(common.average_states(local_states, reference), strict=True)
        functional.reset_net(model)

        if (round_idx + 1) % args.eval_every == 0 or (round_idx + 1) == args.R:
            loss, acc, avg_spikes = evaluate(
                model, data.test_loader, args, functional, neuron.ParametricLIFNode
            )
            best_acc = max(best_acc, acc)
            print(
                f"Round {round_idx + 1} SEW BP/SG Acc@1={acc:.2f}% loss={loss:.4f} "
                f"avg_lif_spikes={avg_spikes:.2f} local_loss={np.mean(local_losses):.4f} "
                f"local_acc={np.mean(local_accs):.2f}% best={best_acc:.2f}%",
                flush=True,
            )
            save_checkpoint(checkpoint_dir / "latest.pt", model, args, round_idx + 1, best_acc)
            if acc >= best_acc:
                save_checkpoint(checkpoint_dir / "best.pt", model, args, round_idx + 1, best_acc)
    print(f"Finished SEW BP/SG. best_acc={best_acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
