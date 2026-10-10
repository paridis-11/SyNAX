#!/usr/bin/env python3
"""Event-consistent SyNAX for Speck neuromorphic datasets.

This runner keeps the SyNAX ANN/SNN client split, but fixes the important
neuromorphic data path: ANN training, Sinabs SNN simulation, and Speck hardware
evaluation all use the same temporal event sequence semantics.
"""

import argparse
import copy
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame


class SpeckNmnistANN(nn.Module):
    """Small CNN that lowers cleanly to DYNAPCNN cores after ANN2SNN."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=4, bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SpeckOfficial128ANN(nn.Module):
    """128x128 adaptation of the official Speck N-MNIST small ANN layout."""

    def __init__(self, num_classes: int, conv_bias: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=2, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=4, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class Speck128ANN(nn.Module):
    """Speck-valid CNN for 128x128 event streams."""

    def __init__(self, num_classes: int, conv_bias: bool = False, variant: str = "small") -> None:
        super().__init__()
        tail_channels = 32
        if variant == "small64":
            tail_channels = 64
        elif variant == "small48":
            tail_channels = 48
        self.features = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, tail_channels, kernel_size=3, stride=2, padding=1, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(tail_channels, num_classes, kernel_size=4, bias=conv_bias),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class Speck128Large64ANN(nn.Module):
    """Larger 9-core Speck candidate for DVS Gesture/CIFAR10-DVS."""

    def __init__(self, num_classes: int, conv_bias: bool = True, variant: str = "large64") -> None:
        super().__init__()
        if variant == "tail64keep":
            channels = [8, 16, 32, 64, 64, 64, 64, 32]
        elif variant == "tail64thin":
            channels = [8, 16, 32, 64, 64, 32, 32, 16]
        elif variant in {"large48tail", "tail48"}:
            channels = [8, 16, 32, 64, 48, 64, 32, 16]
        elif variant in {"large64lite", "tail48lite"}:
            channels = [8, 16, 32, 64, 64, 48, 32, 16]
        elif variant in {"large64m56", "mid56"}:
            channels = [8, 16, 32, 56, 64, 64, 32, 16]
        elif variant in {"large64b56", "bridge56"}:
            channels = [8, 16, 32, 64, 56, 64, 32, 16]
        elif variant in {"large64t56", "tail56"}:
            channels = [8, 16, 32, 64, 64, 56, 32, 16]
        elif variant in {"large64tail24", "tail24"}:
            channels = [8, 16, 32, 64, 64, 64, 24, 16]
        elif variant in {"large64m60", "mid60"}:
            channels = [8, 16, 32, 60, 64, 64, 32, 16]
        elif variant in {"large64b60", "bridge60"}:
            channels = [8, 16, 32, 64, 60, 64, 32, 16]
        elif variant in {"large64t60", "tail60"}:
            channels = [8, 16, 32, 64, 64, 60, 32, 16]
        elif variant in {"large64tail40", "tail40"}:
            channels = [8, 16, 32, 64, 64, 64, 40, 16]
        elif variant in {"mid48", "balanced48"}:
            channels = [8, 16, 32, 48, 64, 64, 32, 16]
        else:
            channels = [8, 16, 32, 64, 64, 64, 32, 16]
        strides = [2, 2, 2, 2, 2, 1, 1, 1]
        layers: List[nn.Module] = []
        in_channels = 2
        for out_channels, stride in zip(channels, strides):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=conv_bias))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.extend(
            [
                nn.Conv2d(in_channels, num_classes, kernel_size=4, bias=conv_bias),
                nn.ReLU(inplace=True),
                nn.Flatten(),
            ]
        )
        self.features = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class IndexDataset(Dataset):
    def __init__(self, base: Dataset, indices: Iterable[int]) -> None:
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        return self.base[self.indices[item]]


@dataclass
class EventData:
    train_loader: List[DataLoader]
    test_loader: DataLoader
    num_classes: int
    input_shape: Tuple[int, int, int]


class EventNPZToFrameDataset(Dataset):
    def __init__(self, root: str, split: str, time_steps: int, sensor_size: Tuple[int, int, int], max_samples: int = 0) -> None:
        self.time_steps = time_steps
        self.sensor_size = sensor_size
        base = Path(root) / split
        if not base.exists():
            raise FileNotFoundError(f"Missing split directory: {base}")
        class_dirs = [path for path in base.iterdir() if path.is_dir()]
        raw_labels = sorted(int(path.name) for path in class_dirs)
        offset = 1 if raw_labels and min(raw_labels) == 1 else 0
        self.samples = []
        for path in sorted(class_dirs, key=lambda item: int(item.name)):
            label = int(path.name) - offset
            self.samples.extend((str(file_path), label) for file_path in sorted(path.glob("*.npz")))
        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        events = self._load_events(path)
        frames = self._events_to_frames(events)
        return torch.from_numpy(frames), label

    @staticmethod
    def _load_events(path: str) -> np.ndarray:
        data = np.load(path)
        if {"x", "y", "t", "p"}.issubset(set(data.files)):
            return np.stack([data["x"], data["y"], data["t"], data["p"]], axis=1).astype(np.float32)
        key = "events" if "events" in data.files else data.files[0]
        events = data[key]
        if events.dtype.fields:
            return np.stack([events["x"], events["y"], events["t"], events["p"]], axis=1).astype(np.float32)
        return events.astype(np.float32)

    def _events_to_frames(self, events: np.ndarray) -> np.ndarray:
        return self._events_to_frames_static(events, self.time_steps, self.sensor_size)

    @staticmethod
    def _events_to_frames_static(events: np.ndarray, time_steps: int, sensor_size: Tuple[int, int, int]) -> np.ndarray:
        width, height, channels = sensor_size
        frames = np.zeros((time_steps, channels, height, width), dtype=np.float32)
        if events.size == 0:
            return frames
        x = np.clip(events[:, 0].astype(np.int64), 0, width - 1)
        y = np.clip(events[:, 1].astype(np.int64), 0, height - 1)
        t = events[:, 2].astype(np.float32)
        p = events[:, 3].astype(np.int64)
        if len(p) and p.min() < 0:
            p = (p + 1) // 2
        p = np.clip(p, 0, channels - 1)
        duration = float(t.max() - t.min())
        if duration <= 0:
            bins = np.zeros_like(x)
        else:
            bins = np.floor((t - t.min()) / (duration + 1e-6) * time_steps).astype(np.int64)
            bins = np.clip(bins, 0, time_steps - 1)
        np.add.at(frames, (bins, p, y, x), 1.0)
        return frames


class FrameNPZDataset(Dataset):
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, root: str, time_steps: int, train: bool, max_samples: int = 0) -> None:
        base = Path(root) / f"frames_number_{time_steps}_split_by_number"
        if not base.exists():
            raise FileNotFoundError(f"Missing frame directory: {base}")
        self.samples = []
        for label, class_name in enumerate(self.CLASSES):
            files = sorted((base / class_name).glob("*.npz"))
            split = int(len(files) * 0.9)
            chosen = files[:split] if train else files[split:]
            self.samples.extend((str(path), label) for path in chosen)
        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        data = np.load(path)
        frames = data["frames" if "frames" in data.files else data.files[0]].astype(np.float32)
        return torch.from_numpy(frames), label


class ClassEventNPZDataset(Dataset):
    CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, root: str, time_steps: int, sensor_size: Tuple[int, int, int], train: bool, max_samples: int = 0) -> None:
        self.time_steps = time_steps
        self.sensor_size = sensor_size
        base = Path(root) / "events_np"
        if not base.exists():
            raise FileNotFoundError(f"Missing raw event directory: {base}")
        self.samples = []
        for label, class_name in enumerate(self.CLASSES):
            files = sorted((base / class_name).glob("*.npz"))
            split = int(len(files) * 0.9)
            chosen = files[:split] if train else files[split:]
            self.samples.extend((str(path), label) for path in chosen)
        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        events, label = self.raw_item(idx)
        frames = EventNPZToFrameDataset._events_to_frames_static(events, self.time_steps, self.sensor_size)
        return torch.from_numpy(frames), label

    def raw_item(self, idx: int):
        path, label = self.samples[idx]
        return EventNPZToFrameDataset._load_events(path), label


class SplitEventNPZRawDataset(Dataset):
    def __init__(self, root: str, split: str, max_samples: int = 0) -> None:
        base = Path(root) / split
        if not base.exists():
            raise FileNotFoundError(f"Missing split directory: {base}")
        class_dirs = [path for path in base.iterdir() if path.is_dir()]
        raw_labels = sorted(int(path.name) for path in class_dirs)
        offset = 1 if raw_labels and min(raw_labels) == 1 else 0
        self.samples = []
        for path in sorted(class_dirs, key=lambda item: int(item.name)):
            label = int(path.name) - offset
            self.samples.extend((str(file_path), label) for file_path in sorted(path.glob("*.npz")))
        if max_samples and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return EventNPZToFrameDataset._load_events(path), label


def dataset_spec(args) -> Tuple[int, Tuple[int, int, int]]:
    name = args.dataset.lower()
    if name == "nmnist":
        return 10, NMNIST.sensor_size
    if name == "dvsgesture":
        return 11, (128, 128, 2)
    if name in {"cifar10dvs", "dvs-cifar10", "cifar-dvs"}:
        return 10, (128, 128, 2)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def build_model(args, num_classes: int) -> nn.Module:
    arch = getattr(args, "arch", "small").lower()
    if tuple(args.input_shape)[1] <= 34:
        return SpeckNmnistANN(num_classes=num_classes)
    if arch in {"official128", "official_small", "official"}:
        return SpeckOfficial128ANN(num_classes=num_classes, conv_bias=bool(getattr(args, "conv_bias", 0)))
    if arch in {"small", "small64", "small48", "small_bias"}:
        return Speck128ANN(num_classes=num_classes, conv_bias=bool(getattr(args, "conv_bias", 0)), variant=arch)
    if arch in {
        "large64",
        "speck128_large64",
        "tail64keep",
        "tail64thin",
        "large48tail",
        "tail48",
        "large64lite",
        "tail48lite",
        "large64m56",
        "mid56",
        "large64b56",
        "bridge56",
        "large64t56",
        "tail56",
        "large64tail24",
        "tail24",
        "large64m60",
        "mid60",
        "large64b60",
        "bridge60",
        "large64t60",
        "tail60",
        "large64tail40",
        "tail40",
        "mid48",
        "balanced48",
    }:
        return Speck128Large64ANN(num_classes=num_classes, conv_bias=bool(getattr(args, "conv_bias", 1)), variant=arch)
    return Speck128ANN(num_classes=num_classes, conv_bias=bool(getattr(args, "conv_bias", 0)), variant=arch)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def subset_indices(length: int, max_samples: int) -> List[int]:
    return list(range(min(length, max_samples))) if max_samples and max_samples > 0 else list(range(length))


def partition_iid(indices: List[int], node_num: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    shuffled = indices[:]
    rng.shuffle(shuffled)
    parts = [[] for _ in range(node_num)]
    for idx, sample_idx in enumerate(shuffled):
        parts[idx % node_num].append(sample_idx)
    return parts


def build_data(args) -> EventData:
    num_classes, sensor_size = dataset_spec(args)
    args.input_shape = (sensor_size[2], sensor_size[1], sensor_size[0])
    name = args.dataset.lower()
    if name == "nmnist":
        transform = ToFrame(sensor_size=sensor_size, n_time_bins=args.time_steps)
        train_set = NMNIST(save_to=args.data_root, train=True, transform=transform)
        test_set = NMNIST(save_to=args.data_root, train=False, transform=transform)
        train_indices = subset_indices(len(train_set), args.max_train_samples)
        test_indices = subset_indices(len(test_set), args.max_test_samples)
    elif name == "dvsgesture":
        train_set = EventNPZToFrameDataset(args.data_root, "train", args.time_steps, sensor_size, args.max_train_samples)
        test_set = EventNPZToFrameDataset(args.data_root, "test", args.time_steps, sensor_size, args.max_test_samples)
        train_indices = list(range(len(train_set)))
        test_indices = list(range(len(test_set)))
    else:
        train_set = ClassEventNPZDataset(args.data_root, args.time_steps, sensor_size, train=True, max_samples=args.max_train_samples)
        test_set = ClassEventNPZDataset(args.data_root, args.time_steps, sensor_size, train=False, max_samples=args.max_test_samples)
        train_indices = list(range(len(train_set)))
        test_indices = list(range(len(test_set)))
    client_indices = partition_iid(train_indices, args.node_num, args.seed)
    loaders = [
        DataLoader(
            IndexDataset(train_set, part),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            pin_memory=True,
        )
        for part in client_indices
    ]
    test_loader = DataLoader(
        Subset(test_set, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    return EventData(loaders, test_loader, num_classes=num_classes, input_shape=args.input_shape)


def flatten_time(data: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, int, int]:
    # tonic ToFrame returns [B,T,C,H,W].
    batch_size, time_steps = data.shape[0], data.shape[1]
    return data.reshape(batch_size * time_steps, *data.shape[2:]).float().to(device), batch_size, time_steps


def temporal_logits(model: nn.Module, data: torch.Tensor, device: torch.device) -> torch.Tensor:
    frames, batch_size, time_steps = flatten_time(data, device)
    logits = model(frames)
    return logits.view(batch_size, time_steps, -1).sum(dim=1)


def build_optimizer(model: nn.Module, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def tensor_delta(new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: new_state[key].detach().float().cpu() - old_state[key].detach().float().cpu()
        for key in new_state
        if key in old_state and new_state[key].is_floating_point()
    }


def average_updates(updates: List[Dict[str, torch.Tensor]], ref_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {key: torch.zeros_like(value, dtype=torch.float32, device="cpu") for key, value in ref_state.items() if value.is_floating_point()}
    if not updates:
        return out
    for update in updates:
        for key, value in update.items():
            if key in out:
                out[key].add_(value)
    for key in out:
        out[key].div_(len(updates))
    return out


def apply_update(model: nn.Module, update: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
    with torch.no_grad():
        params = dict(model.named_parameters())
        for key, value in update.items():
            if key in params:
                params[key].add_(value.to(params[key].device, dtype=params[key].dtype), alpha=scale)


def train_ann_client(model: nn.Module, loader: DataLoader, args, g_t: Dict[str, torch.Tensor], gamma: float, round_idx: int, node_id: int) -> None:
    model.train().to(args.device)
    optimizer = build_optimizer(model, args)
    loss_fn = nn.CrossEntropyLoss()
    g_t_device = {key: value.to(args.device) for key, value in g_t.items()}
    for epoch in range(args.E):
        correct = 0
        total = 0
        last_loss = 0.0
        for batch_idx, (data, target) in enumerate(loader):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            target = target.long().to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = temporal_logits(model, data, args.device)
            loss = loss_fn(logits, target)
            loss.backward()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in g_t_device:
                        param.grad.mul_(args.zero_momentum).add_(g_t_device[name].to(param.grad.dtype), alpha=(1 - args.zero_momentum) * gamma)
            optimizer.step()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.numel()
            last_loss = loss.item()
        print(f"R{round_idx + 1} ANN node{node_id} epoch{epoch + 1}: loss={last_loss:.4f}, acc={100.0 * correct / max(total, 1):.2f}%")


@torch.no_grad()
def evaluate_ann(model: nn.Module, loader: DataLoader, args) -> float:
    model.eval().to(args.device)
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(loader):
        if args.max_eval_batches > 0 and batch_idx >= args.max_eval_batches:
            break
        target = target.long().to(args.device, non_blocking=True)
        logits = temporal_logits(model, data, args.device)
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.numel()
    return 100.0 * correct / max(total, 1)


def convert_ann_to_snn(ann_model: nn.Module, args) -> nn.Module:
    from sinabs.from_torch import from_model

    model = copy.deepcopy(ann_model).to(args.device).eval()
    network = from_model(
        model,
        input_shape=tuple(args.input_shape),
        num_timesteps=args.time_steps,
        spike_threshold=torch.tensor(float(args.spike_threshold), device=args.device),
        min_v_mem=float(args.min_v_mem),
        add_spiking_output=bool(args.add_spiking_output),
    )
    snn = network.spiking_model if hasattr(network, "spiking_model") else network
    return snn.to(args.device).eval()


def reset_snn(model: nn.Module) -> None:
    try:
        import sinabs

        sinabs.reset_states(model)
    except Exception:
        for module in model.modules():
            fn = getattr(module, "reset_states", None)
            if callable(fn):
                fn()


def register_spike_event_hooks(model: nn.Module):
    """Count emitted events from every Sinabs spiking-neuron layer."""
    counter = {"events": 0.0, "layers": 0}
    handles = []

    def count_events(_module, _inputs, output):
        tensor = output[0] if isinstance(output, (tuple, list)) else output
        if torch.is_tensor(tensor):
            counter["events"] += float(tensor.detach().sum().item())

    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if "iaf" in class_name or "lif" in class_name:
            handles.append(module.register_forward_hook(count_events))
            counter["layers"] += 1
    if not handles:
        raise RuntimeError("No Sinabs spiking-neuron layers found for event counting")
    return counter, handles


@torch.no_grad()
def evaluate_snn(model: nn.Module, loader: DataLoader, args) -> Tuple[float, float]:
    model.eval().to(args.device)
    correct = 0
    total = 0
    spike_sum = 0.0
    event_counter, handles = register_spike_event_hooks(model)
    try:
        for batch_idx, (data, target) in enumerate(loader):
            if args.max_eval_batches > 0 and batch_idx >= args.max_eval_batches:
                break
            target = target.long().to(args.device, non_blocking=True)
            frames, batch_size, time_steps = flatten_time(data, args.device)
            reset_snn(model)
            output = model(frames).view(batch_size, time_steps, -1).sum(dim=1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.numel()
            spike_sum += output.sum().item()
    finally:
        for handle in handles:
            handle.remove()
    reset_snn(model)
    model._last_avg_total_spikes = event_counter["events"] / max(total, 1)
    model._last_spike_layer_count = event_counter["layers"]
    return 100.0 * correct / max(total, 1), spike_sum / max(total, 1)


def compute_snn_guidance(current_snn: nn.Module, previous_snn: nn.Module) -> Dict[str, torch.Tensor]:
    curr = current_snn.state_dict()
    prev = previous_snn.state_dict()
    guidance = {}
    flat = []
    for key, value in curr.items():
        if value.is_floating_point() and key in prev and value.shape == prev[key].shape and not key.endswith(".v_mem"):
            diff = value.detach().float() - prev[key].detach().float()
            guidance[key] = diff
            flat.append(diff.flatten())
    if not flat:
        return guidance
    norm = torch.norm(torch.cat(flat))
    return {key: value / norm for key, value in guidance.items()} if norm > 0 else guidance


def compute_snn_feedback(snn_model: nn.Module, loader: DataLoader, guidance: Dict[str, torch.Tensor], args) -> float:
    if not guidance:
        return 0.0
    normalized = {key: value.to(args.device) for key, value in guidance.items()}
    norm_sq = sum(torch.sum(value ** 2) for value in normalized.values())
    if norm_sq == 0:
        return 0.0
    normalized = {key: value / torch.sqrt(norm_sq) for key, value in normalized.items()}
    batches = []
    for idx, batch in enumerate(loader):
        if idx >= args.feedback_batches:
            break
        batches.append(batch)

    def perturb(sign: float) -> nn.Module:
        model = copy.deepcopy(snn_model).to(args.device)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in normalized:
                    param.add_(normalized[name].to(param.device, dtype=param.dtype), alpha=sign * args.zo_epsilon)
        return model

    @torch.no_grad()
    def loss_of(model: nn.Module) -> float:
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        total = 0
        model.eval()
        for data, target in batches:
            target = target.long().to(args.device, non_blocking=True)
            frames, batch_size, time_steps = flatten_time(data, args.device)
            reset_snn(model)
            output = model(frames).view(batch_size, time_steps, -1).sum(dim=1)
            loss = loss_fn(output, target)
            total_loss += loss.item() * target.numel()
            total += target.numel()
        reset_snn(model)
        return total_loss / max(total, 1)

    plus = perturb(1.0)
    loss_plus = loss_of(plus)
    del plus
    minus = perturb(-1.0)
    loss_minus = loss_of(minus)
    del minus
    alpha = (loss_plus - loss_minus) / (2 * args.zo_epsilon)
    if not (math.isfinite(loss_plus) and math.isfinite(loss_minus) and math.isfinite(alpha)):
        print(f"SNN feedback: loss+={loss_plus:.4f}, loss-={loss_minus:.4f}, alpha=nan -> skipped")
        return 0.0
    print(f"SNN feedback: loss+={loss_plus:.4f}, loss-={loss_minus:.4f}, alpha={alpha:.6f}")
    return alpha


def dynapcnn_check(snn_model: nn.Module, args) -> Dict[str, object]:
    if not args.dynapcnn_check:
        return {"backend": "skipped", "ok": True}
    try:
        from sinabs.backend.dynapcnn import DynapcnnNetwork

        dyn = DynapcnnNetwork(copy.deepcopy(snn_model).cpu().eval(), input_shape=tuple(args.input_shape), discretize=True, dvs_input=False)
        return {"backend": "dynapcnn", "ok": True, "layers": len(getattr(dyn, "sequence", []))}
    except Exception as exc:
        return {"backend": "dynapcnn", "ok": False, "error": repr(exc)}


def save_checkpoint(path: str, round_idx: int, args, ann_model: nn.Module, snn_model: nn.Module, ann_acc: float, snn_acc: float, avg_spikes: float, speck_check: Dict[str, object]) -> None:
    payload = {
        "round": round_idx,
        "args": vars(args).copy(),
        "ann_model": cpu_state_dict(ann_model),
        "snn_model": cpu_state_dict(snn_model),
        "ann_acc": ann_acc,
        "snn_acc": snn_acc,
        "avg_output_spikes": avg_spikes,
        "avg_total_spikes": float(getattr(snn_model, "_last_avg_total_spikes", float("nan"))),
        "spike_layers": int(getattr(snn_model, "_last_spike_layer_count", 0)),
        "speck_check": speck_check,
    }
    payload["args"]["device"] = str(payload["args"]["device"])
    torch.save(payload, path)


def hardware_proxy_score(snn_acc: float, avg_spikes: float, args) -> float:
    excess_spikes = max(0.0, avg_spikes - float(args.spike_budget))
    return float(snn_acc) - float(args.spike_penalty) * excess_spikes


def load_event_checkpoint(path: str, device: torch.device) -> Tuple[Dict[str, object], nn.Module, nn.Module]:
    ckpt = torch.load(path, map_location=device)
    args_dict = ckpt.get("args", {})
    num_classes, sensor_size = dataset_spec(argparse.Namespace(dataset=args_dict.get("dataset", "nmnist")))
    ns = argparse.Namespace(
        dataset=args_dict.get("dataset", "nmnist"),
        arch=args_dict.get("arch", "small"),
        conv_bias=int(args_dict.get("conv_bias", 1)),
        time_steps=int(args_dict.get("time_steps", 8)),
        spike_threshold=float(args_dict.get("spike_threshold", 1.0)),
        min_v_mem=float(args_dict.get("min_v_mem", -1.0)),
        add_spiking_output=int(args_dict.get("add_spiking_output", 0)),
        device=device,
        input_shape=tuple(args_dict.get("input_shape", (sensor_size[2], sensor_size[1], sensor_size[0]))),
    )
    model = build_model(ns, num_classes).to(device)
    model.load_state_dict(ckpt["ann_model"], strict=False)
    snn = convert_ann_to_snn(model, ns).to(device)
    if "snn_model" in ckpt:
        state = {key: value for key, value in ckpt["snn_model"].items() if not key.endswith(".v_mem")}
        snn.load_state_dict(state, strict=False)
    return ckpt, model, snn


def make_samna_events(events, spike_cls):
    stream = []
    if isinstance(events, np.ndarray) and events.dtype.fields:
        arr = np.stack([events["x"], events["y"], events["t"], events["p"]], axis=1)
    elif isinstance(events, np.ndarray):
        arr = events
    else:
        arr = np.stack([events["x"], events["y"], events["t"], events["p"]], axis=1)
    if len(arr) == 0:
        return stream
    t0 = int(arr[0, 2])
    for ev in arr:
        spk = spike_cls()
        spk.x = int(ev[0])
        spk.y = int(ev[1])
        spk.timestamp = int(ev[2]) - t0
        spk.feature = int(ev[3])
        spk.layer = 0
        stream.append(spk)
    return stream


def hardware_eval(args) -> None:
    import samna
    from sinabs.backend.dynapcnn import DynapcnnNetwork

    ckpt, _, snn = load_event_checkpoint(args.checkpoint, torch.device("cpu"))
    args_dict = ckpt.get("args", {})
    dyn = DynapcnnNetwork(snn=snn, input_shape=tuple(args_dict.get("input_shape", (2, 34, 34))), discretize=True, dvs_input=False)
    net = dyn.to(device=args.speck_device, chip_layers_ordering="auto")
    print(json.dumps({"checkpoint_round": ckpt.get("round"), "ann_acc": ckpt.get("ann_acc"), "snn_acc": ckpt.get("snn_acc"), "device": args.speck_device, "chip_layers_ordering": list(net.chip_layers_ordering)}, indent=2), flush=True)
    dataset_name = args_dict.get("dataset", "nmnist").lower()
    num_classes, _ = dataset_spec(argparse.Namespace(dataset=dataset_name))
    if dataset_name == "nmnist":
        raw_ds = NMNIST(save_to=args.data_root, train=False)
    elif dataset_name == "dvsgesture":
        raw_ds = SplitEventNPZRawDataset(args.data_root, "test")
    elif dataset_name in {"cifar10dvs", "dvs-cifar10", "cifar-dvs"}:
        raw_ds = ClassEventNPZDataset(args.data_root, int(args_dict.get("time_steps", 8)), (128, 128, 2), train=False)
    else:
        raise ValueError(f"Unsupported hardware dataset: {dataset_name}")
    indices = list(range(0, len(raw_ds), args.hw_stride))[: args.hw_samples]
    correct = 0
    total_events = 0
    spike_cls = samna.speck2f.event.Spike
    for idx in tqdm(indices, desc="hardware-eval"):
        events, label = raw_ds.raw_item(idx) if hasattr(raw_ds, "raw_item") else raw_ds[idx]
        net.reset_states()
        out = net(make_samna_events(events, spike_cls))
        features = [int(each.feature) for each in out if 0 <= int(each.feature) < num_classes]
        pred = Counter(features).most_common(1)[0][0] if features else -1
        correct += int(pred == int(label))
        total_events += len(out)
        print(json.dumps({"idx": idx, "label": int(label), "prediction": int(pred), "output_events": len(out)}), flush=True)
    seen = len(indices)
    print(json.dumps({"samples": seen, "hardware_acc": 100.0 * correct / max(seen, 1), "avg_output_events": total_events / max(seen, 1)}, indent=2), flush=True)
    for name in ("device_input_graph", "device_output_graph"):
        graph = getattr(net, name, None)
        if graph is not None:
            graph.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Event-consistent SyNAX for Speck neuromorphic datasets")
    parser.add_argument("--mode", choices=["train", "hardware"], default="train")
    parser.add_argument("--dataset", default="nmnist", choices=["nmnist", "dvsgesture", "cifar10dvs", "dvs-cifar10", "cifar-dvs"])
    parser.add_argument("--data-root", default="/data_a/lsy/datasets/NMNIST")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speck-device", default="speck2fdevkit:1")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-num", type=int, default=8)
    parser.add_argument("--R", type=int, default=10)
    parser.add_argument("--E", type=int, default=1)
    parser.add_argument("--sample-ratio", type=float, default=0.5)
    parser.add_argument("--snn-ratio", type=float, default=0.5)
    parser.add_argument(
        "--arch",
        default="small",
        choices=[
            "small",
            "small64",
            "small48",
            "small_bias",
            "official128",
            "official_small",
            "official",
            "large64",
            "speck128_large64",
            "tail64keep",
            "tail64thin",
            "large48tail",
            "tail48",
            "large64lite",
            "tail48lite",
            "large64m56",
            "mid56",
            "large64b56",
            "bridge56",
            "large64t56",
            "tail56",
            "large64tail24",
            "tail24",
            "large64m60",
            "mid60",
            "large64b60",
            "bridge60",
            "large64t60",
            "tail60",
            "large64tail40",
            "tail40",
            "mid48",
            "balanced48",
        ],
    )
    parser.add_argument("--conv-bias", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--spike-threshold", type=float, default=1.0)
    parser.add_argument("--min-v-mem", type=float, default=-1.0)
    parser.add_argument("--add-spiking-output", type=int, default=0)
    parser.add_argument("--zo-epsilon", type=float, default=0.01)
    parser.add_argument("--zero-min", type=float, default=-0.5)
    parser.add_argument("--zero-max", type=float, default=0.5)
    parser.add_argument("--zero-momentum", type=float, default=1.0)
    parser.add_argument("--feedback-batches", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--dynapcnn-check", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--hw-samples", type=int, default=100)
    parser.add_argument("--hw-stride", type=int, default=100)
    parser.add_argument("--spike-budget", type=float, default=85.0)
    parser.add_argument("--spike-penalty", type=float, default=0.15)
    parser.add_argument("--proxy-min-acc", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "hardware":
        hardware_eval(args)
        return

    set_seed(args.seed)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data = build_data(args)
    run_name = args.run_name or f"synax_speck_event_{args.dataset}_T{args.time_steps}_b{args.batch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
    checkpoint_dir = os.path.join(args.save_dir, run_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"Running Event-SyNAX on {args.device}: dataset={args.dataset}, arch={args.arch}, input_shape={args.input_shape}, T={args.time_steps}, batch={args.batch_size}, checkpoint_dir={checkpoint_dir}")

    global_model = build_model(args, data.num_classes).to(args.device)
    global_snn = convert_ann_to_snn(global_model, args)
    previous_snn = copy.deepcopy(global_snn)
    G_t = compute_snn_guidance(global_snn, previous_snn)
    g_t = {key: torch.zeros_like(value, dtype=torch.float32, device="cpu") for key, value in global_model.state_dict().items() if value.is_floating_point()}
    print("Initial Speck check:", json.dumps(dynapcnn_check(global_snn, args), ensure_ascii=False))

    ann_count = args.node_num - int(args.node_num * args.snn_ratio)
    device_types = ["ANN" if idx < ann_count else "SNN" for idx in range(args.node_num)]
    best_snn = -1.0
    best_ann = -1.0
    best_proxy = -float("inf")
    best_low_spikes = float("inf")
    for round_idx in range(args.R):
        print(f"=============== Event-SyNAX Round {round_idx + 1}/{args.R} ===============")
        sampled = random.sample(range(args.node_num), max(1, int(args.node_num * args.sample_ratio)))
        ann_nodes = [idx for idx in sampled if device_types[idx] == "ANN"]
        snn_nodes = [idx for idx in sampled if device_types[idx] == "SNN"]
        print(f"sampled={sampled}, ann_nodes={ann_nodes}, snn_nodes={snn_nodes}")

        updates = []
        for node_idx in ann_nodes:
            local = copy.deepcopy(global_model).to(args.device)
            old = copy.deepcopy(local.state_dict())
            steps = max(1, args.E * len(data.train_loader[node_idx]))
            gamma = 1.0 / max(steps * args.lr, 1e-12)
            train_ann_client(local, data.train_loader[node_idx], args, g_t, gamma, round_idx, node_idx + 1)
            updates.append(tensor_delta(local.state_dict(), old))
            del local
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_update = average_updates(updates, global_model.state_dict())
        alphas = [compute_snn_feedback(global_snn, data.train_loader[node_idx], G_t, args) for node_idx in snn_nodes]
        avg_alpha = sum(alphas) / len(alphas) if alphas else 0.0
        clamped_alpha = max(args.zero_min, min(args.zero_max, avg_alpha))
        print(f"Aggregated alpha: {avg_alpha:.6f} -> {clamped_alpha:.6f}")

        old_global = copy.deepcopy(global_model.state_dict())
        apply_update(global_model, avg_update, scale=1.0)
        if clamped_alpha != 0:
            apply_update(global_model, g_t, scale=-clamped_alpha)
        g_t = tensor_delta(old_global, global_model.state_dict())

        previous_snn = copy.deepcopy(global_snn)
        global_snn = convert_ann_to_snn(global_model, args)
        G_t = compute_snn_guidance(global_snn, previous_snn)

        ann_acc = evaluate_ann(global_model, data.test_loader, args)
        print(f"Round {round_idx + 1} ANN Acc: {ann_acc:.2f}%")
        if ann_acc > best_ann:
            best_ann = ann_acc
            torch.save({"round": round_idx + 1, "args": vars(args).copy(), "ann_model": cpu_state_dict(global_model), "ann_acc": ann_acc}, os.path.join(checkpoint_dir, "best_ann.pt"))

        if (round_idx + 1) % args.eval_every == 0 or (round_idx + 1) == args.R:
            snn_acc, avg_spikes = evaluate_snn(global_snn, data.test_loader, args)
            speck_check = dynapcnn_check(global_snn, args)
            avg_total_spikes = float(getattr(global_snn, "_last_avg_total_spikes", float("nan")))
            spike_layers = int(getattr(global_snn, "_last_spike_layer_count", 0))
            print(
                f"Round {round_idx + 1} Sinabs SNN Acc: {snn_acc:.2f}% | "
                f"avg_output_spikes={avg_spikes:.2f} | avg_total_spikes={avg_total_spikes:.2f} | "
                f"spike_layers={spike_layers}"
            )
            print("Speck check:", json.dumps(speck_check, ensure_ascii=False))
            if snn_acc > best_snn:
                best_snn = snn_acc
                save_checkpoint(os.path.join(checkpoint_dir, "best_sinabs_snn.pt"), round_idx + 1, args, global_model, global_snn, ann_acc, snn_acc, avg_spikes, speck_check)
                print(f"Saved best Sinabs SNN checkpoint: round={round_idx + 1}, snn_acc={snn_acc:.2f}%, ann_acc={ann_acc:.2f}%")
            proxy_score = hardware_proxy_score(snn_acc, avg_spikes, args)
            if snn_acc >= args.proxy_min_acc and proxy_score > best_proxy:
                best_proxy = proxy_score
                save_checkpoint(os.path.join(checkpoint_dir, "best_hardware_proxy.pt"), round_idx + 1, args, global_model, global_snn, ann_acc, snn_acc, avg_spikes, speck_check)
                print(f"Saved best hardware-proxy checkpoint: round={round_idx + 1}, score={proxy_score:.2f}, snn_acc={snn_acc:.2f}%, avg_output_spikes={avg_spikes:.2f}")
            if snn_acc >= args.proxy_min_acc and avg_spikes < best_low_spikes:
                best_low_spikes = avg_spikes
                save_checkpoint(os.path.join(checkpoint_dir, "best_low_spike.pt"), round_idx + 1, args, global_model, global_snn, ann_acc, snn_acc, avg_spikes, speck_check)
                print(f"Saved best low-spike checkpoint: round={round_idx + 1}, snn_acc={snn_acc:.2f}%, avg_output_spikes={avg_spikes:.2f}")
    print(f"Finished Event-SyNAX. best_ann={best_ann:.2f}%, best_snn={best_snn:.2f}%, best_proxy={best_proxy:.2f}")


if __name__ == "__main__":
    main()
