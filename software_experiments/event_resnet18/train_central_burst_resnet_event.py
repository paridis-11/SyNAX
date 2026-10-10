#!/usr/bin/env python3
"""Centralized ANN backprop control for the Burst neuromorphic ResNet."""

import argparse
import math
import os
import random
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import synax_neuro as burst


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--burst-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", default="cifar10dvs", choices=["cifar10dvs", "ncaltech101"])
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--augment", type=int, default=0)
    parser.add_argument("--crop-padding", type=int, default=8)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--temporal-shift", type=int, default=2)
    parser.add_argument("--bin-drop-prob", type=float, default=0.1)
    parser.add_argument("--polarity-drop-prob", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--cifar-stem", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


class EventConsistentAugment(Dataset):
    """Apply one spatial/temporal transform consistently to all event channels."""

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
            sample = sample[:, top : top + 128, left : left + 128]
        if random.random() < self.args.hflip_prob:
            sample = torch.flip(sample, dims=(-1,))

        event = sample.reshape(2, 9, sample.shape[-2], sample.shape[-1])
        max_shift = self.args.temporal_shift
        if max_shift > 0:
            shift = random.randint(-max_shift, max_shift)
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


def mix_batch(inputs, targets, args):
    if args.mix_prob <= 0 or random.random() >= args.mix_prob:
        return inputs, targets, targets, 1.0
    choices = []
    if args.mixup_alpha > 0:
        choices.append("mixup")
    if args.cutmix_alpha > 0:
        choices.append("cutmix")
    if not choices:
        return inputs, targets, targets, 1.0

    mode = random.choice(choices)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    if mode == "mixup":
        lam = random.betavariate(args.mixup_alpha, args.mixup_alpha)
        mixed = inputs.mul(lam).add(inputs[index], alpha=1.0 - lam)
    else:
        lam = random.betavariate(args.cutmix_alpha, args.cutmix_alpha)
        height, width = inputs.shape[-2:]
        cut_ratio = math.sqrt(1.0 - lam)
        cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
        center_y, center_x = random.randrange(height), random.randrange(width)
        y1, y2 = max(0, center_y - cut_h // 2), min(height, center_y + cut_h // 2)
        x1, x2 = max(0, center_x - cut_w // 2), min(width, center_x + cut_w // 2)
        mixed = inputs.clone()
        mixed[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(height * width))
    return mixed, targets, targets[index], lam


def data_args(args):
    return SimpleNamespace(
        dataset=args.dataset,
        data_path=args.data_path,
        frame_data=1,
        speck_input=0,
        input_size=128,
        input_channels=2,
        input_scale=1.0,
        node_num=1,
        seed=args.seed,
        partition="iid",
        dir_alpha=0.5,
        batch_size=args.batch_size,
        workers=args.workers,
    )


def run_epoch(model, loader, criterion, optimizer, device, args):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if args.max_train_batches and batch_idx >= args.max_train_batches:
            break
        inputs = inputs.float().to(device, non_blocking=True)
        targets = targets.long().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        inputs, targets_a, targets_b, lam = mix_batch(inputs, targets, args)
        logits = burst.normalize_output(model(inputs))
        loss = lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        correct += logits.argmax(1).eq(targets).sum().item()
        total += batch_size
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, max_batches):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        inputs = inputs.float().to(device, non_blocking=True)
        targets = targets.long().to(device, non_blocking=True)
        logits = burst.normalize_output(model(inputs))
        loss = criterion(logits, targets)
        batch_size = targets.numel()
        total_loss += loss.item() * batch_size
        correct += logits.argmax(1).eq(targets).sum().item()
        total += batch_size
    return total_loss / max(total, 1), 100.0 * correct / max(total, 1)


def main():
    args = parse_args()
    burst.add_burst_paths(args.burst_root)
    burst.set_random_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = burst.build_neuro_data(data_args(args))
    model = burst.build_neuro_model("resnet18", data.num_classes).to(device)
    if args.cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(device)
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
        model.maxpool = nn.Identity()
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_scale(epoch):
        if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
            return float(epoch + 1) / args.warmup_epochs
        denominator = max(1, args.epochs - args.warmup_epochs)
        progress = (epoch - args.warmup_epochs) / denominator
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_scale)
    train_loader = data.train_loader[0]
    if args.augment:
        train_loader = DataLoader(
            EventConsistentAugment(train_loader.dataset, args),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    checkpoint_dir = os.path.join(args.save_dir, args.run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Central ANN BP: dataset={args.dataset} model=burst-resnet18 init=scratch "
        f"params={params} input=18x128x128 epochs={args.epochs} batch={args.batch_size} "
        f"optimizer={args.optimizer} augment={bool(args.augment)} cifar_stem={bool(args.cifar_stem)}"
    )
    print("ANN2SNN=False SNN=False zero_order_feedback=False")

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args,
        )
        scheduler.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            test_loss, test_acc = evaluate(
                model, data.test_loader, criterion, device, args.max_eval_batches
            )
            best_acc = max(best_acc, test_acc)
            print(
                f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"test_loss={test_loss:.4f} Acc@1={test_acc:.2f}% best={best_acc:.2f}% "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )
            if test_acc >= best_acc:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "test_acc": test_acc,
                        "args": vars(args),
                    },
                    os.path.join(checkpoint_dir, "best.pt"),
                )
    print(f"Finished central ANN BP. best_acc={best_acc:.2f}%")


if __name__ == "__main__":
    main()
