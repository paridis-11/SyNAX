"""
SyNAX for neuromorphic datasets.

This runner keeps the SyNAX ANN/SNN client split used by SyNAX.py, but uses the
burst-ann2snn-master neuromorphic pipeline for data, models, and ANN-to-SNN
conversion.  The default conversion is the tested adaptive calibration path:
light calibration + burst max-spike search, without threshold search.
"""

import argparse
import copy
import math
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def add_burst_paths(burst_root: str) -> None:
    burst_root = os.path.abspath(burst_root)
    neuromorphic_root = os.path.join(burst_root, "neuromorphic")
    for path in (neuromorphic_root, burst_root):
        if path not in sys.path:
            sys.path.insert(0, path)


@dataclass
class NeuroData:
    train_loader: List[DataLoader]
    test_loader: DataLoader
    calib_loader: DataLoader
    num_classes: int


class EventConsistentAugment(torch.utils.data.Dataset):
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
            size = self.args.input_size
            sample = sample[:, top : top + size, left : left + size]
        if random.random() < self.args.hflip_prob:
            sample = torch.flip(sample, dims=(-1,))

        if sample.shape[0] == 18:
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
            sample = event.reshape_as(sample)
        return sample, target


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
    alpha = args.mixup_alpha if mode == "mixup" else args.cutmix_alpha
    lam = random.betavariate(alpha, alpha)
    if mode == "mixup":
        mixed = inputs.mul(lam).add(inputs[index], alpha=1.0 - lam)
    else:
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


class SpeckInputDataset(torch.utils.data.Dataset):
    """Convert voxel-grid neuromorphic samples to Speck-compatible DVS frames.

    The upstream burst-ann2snn loader emits 18 channels: 9 temporal bins for OFF and
    9 temporal bins for ON events, resized to 224x224. Speck's DVS input exposes
    polarity channels directly, so this wrapper collapses temporal bins to 2 polarity
    channels and resizes samples to 128x128 for DYNAPCNN deployment checks.
    """

    def __init__(self, dataset, input_size: int = 128, channels: int = 2, scale: float = 1.0) -> None:
        self.dataset = dataset
        self.input_size = input_size
        self.channels = channels
        self.scale = scale

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        sample, target = self.dataset[index]
        sample = sample.float()
        if sample.dim() != 3:
            raise ValueError(f"Expected CHW sample, got shape {tuple(sample.shape)}")

        if self.channels == 2 and sample.shape[0] == 18:
            sample = sample.view(2, 9, sample.shape[1], sample.shape[2]).sum(dim=1)
        elif self.channels == 1 and sample.shape[0] == 18:
            sample = sample.sum(dim=0, keepdim=True)
        elif sample.shape[0] != self.channels:
            raise ValueError(f"Cannot convert {sample.shape[0]} channels to {self.channels} Speck channels")

        if sample.shape[-2:] != (self.input_size, self.input_size):
            sample = F.interpolate(
                sample.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if self.scale != 1.0:
            sample = sample / self.scale
        return sample, target


class EventFrameStaticDataset(torch.utils.data.Dataset):
    """Load pre-integrated event frames and expose burst-ann2snn's 18-channel format.

    CIFAR10-DVS/N-Caltech101 frame caches are stored as [T, 2, H, W].  The original
    burst-ann2snn neuromorphic models consume 18 channels: 9 temporal bins for each
    polarity.  This adapter resamples T to 9 and flattens [2, 9, H, W] -> [18, H, W].
    """

    CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, root: str, dataset: str, split: str, input_size: int = 128):
        self.root = Path(root)
        self.dataset = dataset.lower()
        self.split = split
        self.input_size = input_size
        self.samples: List[Tuple[str, int]] = []
        if self.dataset in {"cifar10dvs", "dvs-cifar10", "cifar-dvs"}:
            self._build_cifar10dvs()
        elif self.dataset in {"dvsgesture", "dvs128gesture", "dvs-gesture"}:
            self._build_dvsgesture()
        elif self.dataset in {"ncaltech101", "n-caltech101", "ncaltech"}:
            self._build_ncaltech101()
        else:
            raise ValueError(f"Unsupported frame dataset: {dataset}")

    def _build_cifar10dvs(self) -> None:
        candidates = [
            self.root / "CIFAR10DVS" / "frames_number_16_split_by_number",
            self.root / "frames_number_16_split_by_number",
        ]
        base = next((path for path in candidates if path.exists()), None)
        if base is None:
            raise FileNotFoundError(f"Missing CIFAR10DVS frames under {self.root}")
        for label, class_name in enumerate(self.CIFAR_CLASSES):
            files = sorted((base / class_name).glob("*.npz"))
            cut = int(len(files) * 0.9)
            chosen = files[:cut] if self.split == "train" else files[cut:]
            self.samples.extend((str(path), label) for path in chosen)

    def _build_dvsgesture(self) -> None:
        candidates = [
            self.root / "DvsGesture" / "events_np" / self.split,
            self.root / "events_np" / self.split,
            self.root / "DvsGesture" / self.split,
            self.root / self.split,
        ]
        base = next((path for path in candidates if path.exists()), None)
        if base is None:
            raise FileNotFoundError(f"Missing DVS Gesture events under {self.root}")
        classes = sorted([path for path in base.iterdir() if path.is_dir()], key=lambda p: int(p.name))
        for class_dir in classes:
            label = int(class_dir.name)
            files = sorted(class_dir.glob("*.npz"))
            self.samples.extend((str(path), label) for path in files)

    def _build_ncaltech101(self) -> None:
        candidates = [
            self.root / "NCaltech101" / "frames_number_16_split_by_number",
            self.root / "N-Caltech101" / "frames_number_16_split_by_number",
            self.root / "frames_number_16_split_by_number",
        ]
        base = next((path for path in candidates if path.exists()), None)
        if base is None:
            raise FileNotFoundError(f"Missing N-Caltech101 frames under {self.root}")
        classes = sorted([path for path in base.iterdir() if path.is_dir() and path.name != "BACKGROUND_Google"])
        for label, class_dir in enumerate(classes):
            files = sorted(class_dir.glob("*.npz"))
            cut = int(len(files) * 0.9)
            chosen = files[:cut] if self.split == "train" else files[cut:]
            self.samples.extend((str(path), label) for path in chosen)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        data = np.load(path)
        if "frames" in data.files:
            frames = data["frames"].astype(np.float32)
        elif {"x", "y", "t", "p"}.issubset(set(data.files)):
            frames = self._events_to_frames(data)
        else:
            frames = data[data.files[0]].astype(np.float32)
        if frames.ndim != 4 or frames.shape[1] != 2:
            raise ValueError(f"Expected [T,2,H,W] frames, got {frames.shape} from {path}")
        frames_t = torch.from_numpy(frames).float()
        if frames_t.shape[0] != 9:
            # Interpolate along time independently for every pixel and polarity.
            t, c, h, w = frames_t.shape
            flat = frames_t.permute(1, 2, 3, 0).reshape(1, c * h * w, t)
            flat = F.interpolate(flat, size=9, mode="linear", align_corners=False)
            frames_t = flat.reshape(c, h, w, 9).permute(3, 0, 1, 2)
        if frames_t.shape[-2:] != (self.input_size, self.input_size):
            frames_t = F.interpolate(frames_t, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        voxel = frames_t.permute(1, 0, 2, 3).reshape(18, self.input_size, self.input_size)
        return voxel, int(target)

    def _events_to_frames(self, data) -> np.ndarray:
        frames = np.zeros((9, 2, 128, 128), dtype=np.float32)
        x = np.clip(data["x"].astype(np.int64), 0, 127)
        y = np.clip(data["y"].astype(np.int64), 0, 127)
        t = data["t"].astype(np.float32)
        p = data["p"].astype(np.int64)
        if len(p) and p.min() < 0:
            p = (p + 1) // 2
        p = np.clip(p, 0, 1)
        if len(t) == 0:
            return frames
        duration = float(t.max() - t.min())
        if duration <= 0:
            bins = np.zeros_like(x)
        else:
            bins = np.floor((t - t.min()) / (duration + 1e-6) * 9).astype(np.int64)
            bins = np.clip(bins, 0, 8)
        np.add.at(frames, (bins, p, y, x), 1.0)
        return frames


def labels_of(dataset: torch.utils.data.Dataset) -> List[int]:
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in dataset.samples]
    labels = []
    for idx in range(len(dataset)):
        labels.append(int(dataset[idx][1]))
    return labels


def partition_iid(labels: Sequence[int], node_num: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    indices = list(range(len(labels)))
    rng.shuffle(indices)
    parts = [[] for _ in range(node_num)]
    for pos, idx in enumerate(indices):
        parts[pos % node_num].append(idx)
    return parts


def partition_dirichlet(labels: Sequence[int], node_num: int, alpha: float, seed: int) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    labels_np = np.array(labels)
    parts = [[] for _ in range(node_num)]
    for cls in sorted(set(int(x) for x in labels)):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        props = rng.dirichlet(np.full(node_num, alpha))
        cuts = (np.cumsum(props) * len(cls_idx)).astype(int)[:-1]
        for node, chunk in enumerate(np.split(cls_idx, cuts)):
            parts[node].extend(chunk.tolist())
    for part in parts:
        rng.shuffle(part)
    return parts


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_output(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def tensor_delta(new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    delta = {}
    for key, value in new_state.items():
        if value.is_floating_point() and key in old_state:
            delta[key] = value.detach().float() - old_state[key].detach().float()
    return delta


def average_updates(updates: List[Dict[str, torch.Tensor]], reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not updates:
        return {k: torch.zeros_like(v, dtype=torch.float32) for k, v in reference_state.items() if v.is_floating_point()}

    avg = {}
    for key, value in reference_state.items():
        if not value.is_floating_point():
            continue
        parts = [update[key] for update in updates if key in update]
        if parts:
            avg[key] = sum(parts) / len(parts)
        else:
            avg[key] = torch.zeros_like(value, dtype=torch.float32)
    return avg


def apply_update(model: nn.Module, avg_update: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
    state = model.state_dict()
    new_state = copy.deepcopy(state)
    for key, value in state.items():
        if value.is_floating_point() and key in avg_update:
            update = avg_update[key].to(value.device, dtype=value.dtype)
            if not torch.isfinite(update).all():
                update = torch.nan_to_num(update, nan=0.0, posinf=0.0, neginf=0.0)
            new_state[key] = value + scale * update
    model.load_state_dict(new_state)


def apply_normalized_executor_correction(
    model: nn.Module,
    g_t: Dict[str, torch.Tensor],
    clamped_alpha: float,
    eta_snn: float,
    direction_mode: str = "guided",
    random_seed: int = 0,
) -> float:
    parameter_names = set(dict(model.named_parameters()))
    valid_g = {
        key: value
        for key, value in g_t.items()
        if key in parameter_names and value.is_floating_point()
    }
    norm_sq = sum(torch.sum(value.detach().float() ** 2).item() for value in valid_g.values())
    g_norm = math.sqrt(norm_sq)
    print(
        f"Executor correction: g_norm={g_norm:.8e}, "
        f"old_effective_step={abs(clamped_alpha) * g_norm:.8e}, "
        f"new_effective_step={abs(eta_snn * clamped_alpha):.8e}",
        flush=True,
    )
    if g_norm <= 1e-12 or clamped_alpha == 0:
        return g_norm
    if direction_mode == "random":
        random_direction = {}
        random_norm_sq = 0.0
        generators = {}
        for key, value in valid_g.items():
            device = value.device
            device_key = str(device)
            if device_key not in generators:
                generators[device_key] = torch.Generator(device=device).manual_seed(random_seed)
            noise = torch.randn(
                value.shape,
                generator=generators[device_key],
                device=device,
                dtype=torch.float32,
            )
            random_direction[key] = noise
            random_norm_sq += torch.sum(noise ** 2).item()
        random_norm = math.sqrt(random_norm_sq)
        if random_norm <= 1e-12:
            return g_norm
        direction = {key: value / random_norm for key, value in random_direction.items()}
        print(
            f"Executor direction: random matched-unit-norm seed={random_seed}",
            flush=True,
        )
    else:
        direction = {key: value / g_norm for key, value in valid_g.items()}
        print("Executor direction: guided", flush=True)
    apply_update(model, direction, scale=-(eta_snn * clamped_alpha))
    return g_norm


def build_neuro_model(arch: str, num_classes: int) -> nn.Module:
    from models.resnet import spiking_resnet18, spiking_resnet34, spiking_resnet50, spiking_resnet101
    from models.vgg import vgg11, vgg16
    try:
        from models.vit import vit_cifar10dvs, vit_tiny_cifar10dvs
    except ImportError:
        vit_cifar10dvs = None
        vit_tiny_cifar10dvs = None
    try:
        from models.speck_cnn import speck_cnn, speck_cnn_large, speck_cnn_tail, speck_cnn_tail64, speck_cnn_xlarge
    except ImportError:
        from speck_cnn import speck_cnn, speck_cnn_large, speck_cnn_tail, speck_cnn_tail64, speck_cnn_xlarge

    arch = arch.lower()
    if arch == "cnn":
        raise ValueError("Use a concrete CNN arch: resnet18, resnet34, resnet50, resnet101, vgg11, or vgg16.")
    if arch == "resnet18":
        return spiking_resnet18(num_classes=num_classes)
    if arch == "resnet34":
        return spiking_resnet34(num_classes=num_classes)
    if arch == "resnet50":
        return spiking_resnet50(num_classes=num_classes)
    if arch == "resnet101":
        return spiking_resnet101(num_classes=num_classes)
    if arch == "vgg11":
        return vgg11(num_classes=num_classes)
    if arch == "vgg16":
        return vgg16(num_classes=num_classes)
    if arch == "vit" and vit_cifar10dvs is not None:
        return vit_cifar10dvs(num_classes=num_classes)
    if arch == "vit_tiny" and vit_tiny_cifar10dvs is not None:
        return vit_tiny_cifar10dvs(num_classes=num_classes)
    if arch == "speck_cnn":
        return speck_cnn(num_classes=num_classes)
    if arch == "speck_cnn_large":
        return speck_cnn_large(num_classes=num_classes)
    if arch == "speck_cnn_tail":
        return speck_cnn_tail(num_classes=num_classes)
    if arch == "speck_cnn_tail64":
        return speck_cnn_tail64(num_classes=num_classes)
    if arch == "speck_cnn_xlarge":
        return speck_cnn_xlarge(num_classes=num_classes)
    raise ValueError(f"Unsupported neuromorphic architecture: {arch}")


def get_specials(arch: str):
    from models.resnet import res_spcials
    from models.vgg import vgg_specials

    arch = arch.lower()
    if arch.startswith("resnet"):
        return res_spcials
    if arch.startswith("vgg"):
        return vgg_specials
    return {}


def load_checkpoint_if_needed(model: nn.Module, checkpoint: str, device: torch.device) -> None:
    if not checkpoint:
        return

    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("ann_model", ckpt.get("model", ckpt))
    clean_state = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "base_model.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        clean_state[new_key] = value
    model.load_state_dict(clean_state, strict=True)
    print(f"Loaded ANN checkpoint: {checkpoint}")


def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def serialize_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return None


def collect_snn_runtime_state(model: nn.Module) -> Dict[str, Dict[str, object]]:
    runtime_state = {}
    for module_name, module in model.named_modules():
        attrs = {}
        for attr_name in ("threshold", "max", "maxspike", "sim_length", "use_spike", "enable_shift"):
            if hasattr(module, attr_name):
                value = serialize_value(getattr(module, attr_name))
                if value is not None:
                    attrs[attr_name] = value
        if attrs:
            runtime_state[module_name] = attrs
    return runtime_state


def restore_snn_runtime_state(model: nn.Module, runtime_state: Dict[str, Dict[str, object]]) -> None:
    module_lookup = dict(model.named_modules())
    for module_name, attrs in runtime_state.items():
        module = module_lookup.get(module_name)
        if module is None:
            continue
        device = next(module.parameters(), torch.empty(0)).device
        for attr_name, value in attrs.items():
            if torch.is_tensor(value):
                value = value.to(device)
            setattr(module, attr_name, value)


def save_training_checkpoint(
    path: str,
    *,
    round_idx: int,
    args,
    ann_model: nn.Module,
    snn_model: nn.Module = None,
    ann_acc: float = None,
    snn_acc: float = None,
    spikecount: float = None,
) -> None:
    payload = {
        "round": round_idx,
        "arch": args.arch,
        "dataset": args.dataset,
        "ann_model": cpu_state_dict(ann_model),
        "ann_acc": ann_acc,
        "snn_acc": snn_acc,
        "spikecount": spikecount,
        "args": vars(args).copy(),
    }
    payload["args"]["device"] = str(payload["args"]["device"])
    if snn_model is not None:
        payload["snn_model"] = cpu_state_dict(snn_model)
        payload["snn_runtime"] = collect_snn_runtime_state(snn_model)
    torch.save(payload, path)


def build_neuro_data(args) -> NeuroData:
    if args.frame_data:
        dataset_name = args.dataset.lower()
        if dataset_name in {"cifar10dvs", "dvs-cifar10", "cifar-dvs"}:
            num_classes = 10
        elif dataset_name in {"dvsgesture", "dvs128gesture", "dvs-gesture"}:
            num_classes = 11
        else:
            num_classes = 101
        trainset = EventFrameStaticDataset(args.data_path, dataset_name, "train", input_size=args.input_size)
        testset = EventFrameStaticDataset(args.data_path, dataset_name, "test", input_size=args.input_size)
        labels = labels_of(trainset)
        if args.partition == "iid":
            parts = partition_iid(labels, args.node_num, args.seed)
        else:
            parts = partition_dirichlet(labels, args.node_num, args.dir_alpha, args.seed)
        splits = [torch.utils.data.Subset(trainset, part) for part in parts]
    else:
        from convert_neuromorphic import load_data

        train_all_loader, test_loader, num_classes = load_data(
            args.dataset,
            args.batch_size,
            args.data_path,
        )
        trainset = train_all_loader.dataset
        testset = test_loader.dataset
        if args.speck_input:
            trainset = SpeckInputDataset(
                trainset,
                input_size=args.input_size,
                channels=args.input_channels,
                scale=args.input_scale,
            )
            testset = SpeckInputDataset(
                testset,
                input_size=args.input_size,
                channels=args.input_channels,
                scale=args.input_scale,
            )
        split_lengths = [len(trainset) // args.node_num] * args.node_num
        split_lengths[-1] += len(trainset) - sum(split_lengths)
        splits = random_split(trainset, split_lengths, generator=torch.Generator().manual_seed(args.seed))

    train_loaders = [
        DataLoader(
            split,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=True,
        )
        for split in splits
    ]
    calib_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        pin_memory=True,
    )
    return NeuroData(train_loaders, test_loader, calib_loader, num_classes)


def add_train_augmentation(data: NeuroData, args) -> None:
    if not args.augment:
        return
    data.train_loader = [
        DataLoader(
            EventConsistentAugment(loader.dataset, args),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )
        for loader in data.train_loader
    ]


def scheduled_lr(initial_lr: float, round_idx: int, rounds: int, warmup_rounds: int) -> float:
    if warmup_rounds > 0 and round_idx < warmup_rounds:
        return initial_lr * float(round_idx + 1) / warmup_rounds
    progress = (round_idx - warmup_rounds) / max(1, rounds - warmup_rounds)
    return initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: nn.Module, args, lr: float):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def train_synax_ann(model: nn.Module, loader: DataLoader, args, g_t: Dict[str, torch.Tensor], gamma: float, lr: float, round_idx: int, node_id: int) -> float:
    model.to(args.device).train()
    optimizer = build_optimizer(model, args, lr)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    g_t_device = {k: v.to(args.device) for k, v in g_t.items()}

    last_loss = 0.0
    for epoch in range(args.E):
        correct = 0
        seen = 0
        progress = tqdm(loader, desc=f"R{round_idx + 1} ANN node{node_id} epoch{epoch + 1}", mininterval=2)
        for batch_idx, (data, target) in enumerate(progress):
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
            data = data.float().to(args.device, non_blocking=True)
            target = target.long().to(args.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            data, target_a, target_b, lam = mix_batch(data, target, args)
            output = normalize_output(model(data))
            loss = lam * loss_fn(output, target_a) + (1.0 - lam) * loss_fn(output, target_b)
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None and name in g_t_device:
                        param.grad.data = args.zero_momentum * param.grad.data + (1 - args.zero_momentum) * gamma * g_t_device[name]

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            last_loss = loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            seen += target.numel()
            progress.set_postfix(loss=f"{last_loss:.4f}", acc=f"{100.0 * correct / max(seen, 1):.2f}")
    return last_loss


@torch.no_grad()
def evaluate_ann(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.to(device).eval()
    correct = 0
    total = 0
    for data, target in loader:
        data = data.float().to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True)
        output = normalize_output(model(data))
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.numel()
    return 100.0 * correct / max(total, 1)


@torch.no_grad()
def evaluate_snn(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    from general_utils.spiking_layer import SpikeModule
    from models.resnet import SpikeResModule

    model.to(device).eval()
    correct = 0
    total = 0
    spikecount = 0.0
    for data, target in loader:
        data = data.float().to(device, non_blocking=True)
        target = target.long().to(device, non_blocking=True)
        output = normalize_output(model(data))
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.numel()
        for module in model.modules():
            if isinstance(module, (SpikeModule, SpikeResModule)):
                spikecount += module.spike_counter
                module.spike_counter = 0
    return 100.0 * correct / max(total, 1), spikecount


def convert_ann_to_burst_snn(ann_model: nn.Module, calib_loader: DataLoader, args) -> nn.Module:
    from general_utils.calibration import bias_corr_model
    from general_utils.fold_bn import search_fold_and_remove_bn
    from general_utils.spiking_layer import SpikeModel, SpikeModule
    from general_utils.spiking_layer import get_maximum_activation, sensitivity_anylysis
    from models.resnet import SpikeResModule

    model = copy.deepcopy(ann_model).to(args.device)
    search_fold_and_remove_bn(model)

    desired_maxspike = args.maxspike
    convert_maxspike = args.initialspike if args.burst_search else args.maxspike
    snn = SpikeModel(
        model=model,
        sim_length=args.test_T,
        maxspike=convert_maxspike,
        specials=get_specials(args.arch),
    ).to(args.device)

    mse = args.calib != "none"
    get_maximum_activation(
        calib_loader,
        model=snn,
        momentum=0.9,
        iters=args.calib_iters,
        mse=mse,
        percentile=None,
        maxspike=convert_maxspike,
        sim_length=args.test_T,
        channel_wise=True,
    )

    # Scratch models can contain inactive channels during the first rounds.
    # Their calibrated threshold is zero, which would make the Burst membrane
    # update divide by zero and poison the zero-order loss with NaNs.
    for module in snn.modules():
        if isinstance(module, (SpikeModule, SpikeResModule)) and torch.is_tensor(module.threshold):
            finite_positive = module.threshold[torch.isfinite(module.threshold) & (module.threshold > 0)]
            fallback = finite_positive.median() if finite_positive.numel() else module.threshold.new_tensor(1.0)
            module.threshold.data.copy_(
                torch.nan_to_num(module.threshold.data, nan=float(fallback), posinf=float(fallback), neginf=1e-4)
                .clamp_min_(1e-4)
            )

    if args.burst_search:
        for _ in range(args.searchtime):
            optimal_maxspike_list, _ = sensitivity_anylysis(
                calib_loader,
                model=snn,
                maxspike=convert_maxspike,
                maxspike_ratio=args.maxspike_ratio,
                sim_length=args.test_T,
                disred_maxspike=desired_maxspike,
                minspike=args.minspike,
                metric=args.metric,
                method=args.search_method,
            )
            index = 0
            for module in snn.modules():
                if isinstance(module, SpikeModule):
                    module.maxspike = optimal_maxspike_list[index]
                    index += 1
            get_maximum_activation(
                calib_loader,
                model=snn,
                momentum=0.9,
                iters=args.calib_iters,
                mse=mse,
                percentile=None,
                maxspike=convert_maxspike,
                sim_length=args.test_T,
                channel_wise=True,
            )
            for module in snn.modules():
                if isinstance(module, (SpikeModule, SpikeResModule)) and torch.is_tensor(module.threshold):
                    finite_positive = module.threshold[torch.isfinite(module.threshold) & (module.threshold > 0)]
                    fallback = finite_positive.median() if finite_positive.numel() else module.threshold.new_tensor(1.0)
                    module.threshold.data.copy_(
                        torch.nan_to_num(module.threshold.data, nan=float(fallback), posinf=float(fallback), neginf=1e-4)
                        .clamp_min_(1e-4)
                    )

    if args.calib == "light":
        bias_corr_model(model=snn, train_loader=calib_loader, correct_mempot=False)

    for module in snn.modules():
        if isinstance(module, (SpikeModule, SpikeResModule)):
            module.spike_counter = 0
    snn.set_spike_state(use_spike=True)
    return snn


def compute_snn_guidance(current_snn: nn.Module, previous_snn: nn.Module) -> Dict[str, torch.Tensor]:
    current_state = current_snn.state_dict()
    previous_state = previous_snn.state_dict()
    guidance = {}
    flat = []
    for key, value in current_state.items():
        if value.is_floating_point() and key in previous_state:
            diff = value.detach().float() - previous_state[key].detach().float()
            guidance[key] = diff
            flat.append(diff.flatten())
    if not flat:
        return guidance
    norm = torch.norm(torch.cat(flat))
    if norm > 0:
        guidance = {key: value / norm for key, value in guidance.items()}
    return guidance


def compute_snn_feedback(snn_model: nn.Module, loader: DataLoader, guidance: Dict[str, torch.Tensor], args) -> float:
    norm_sq = sum(torch.sum(value.to(args.device) ** 2) for value in guidance.values())
    if not guidance or norm_sq == 0:
        return 0.0

    normalized = {key: value.to(args.device) / torch.sqrt(norm_sq) for key, value in guidance.items()}
    data_batches = []
    for idx, batch in enumerate(loader):
        if idx >= args.feedback_batches:
            break
        data_batches.append(batch)

    def apply_perturbation(model: nn.Module, sign: float) -> nn.Module:
        perturbed = copy.deepcopy(model).to(args.device)
        with torch.no_grad():
            for name, param in perturbed.named_parameters():
                if name in normalized:
                    param.add_(sign * args.zo_epsilon * normalized[name].to(param.device, dtype=param.dtype))
        return perturbed

    def get_loss(model: nn.Module) -> float:
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0
        model.eval()
        with torch.no_grad():
            for data, target in data_batches:
                data = data.float().to(args.device, non_blocking=True)
                target = target.long().to(args.device, non_blocking=True)
                output = normalize_output(model(data))
                loss = loss_fn(output, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
        return total_loss / max(total_samples, 1)

    model_plus = apply_perturbation(snn_model, 1.0)
    loss_plus = get_loss(model_plus)
    del model_plus

    model_minus = apply_perturbation(snn_model, -1.0)
    loss_minus = get_loss(model_minus)
    del model_minus

    alpha = (loss_plus - loss_minus) / (2 * args.zo_epsilon)
    if not (math.isfinite(loss_plus) and math.isfinite(loss_minus) and math.isfinite(alpha)):
        print(f"SNN feedback: loss+={loss_plus:.4f}, loss-={loss_minus:.4f}, alpha=nan -> skipped")
        return 0.0
    print(f"SNN feedback: loss+={loss_plus:.4f}, loss-={loss_minus:.4f}, alpha={alpha:.6f}")
    return alpha


def parse_args():
    parser = argparse.ArgumentParser(description="SyNAX on neuromorphic datasets with burst ANN2SNN conversion")
    parser.add_argument("--burst-root", default="/data_a/lsy/code/burst-ann2snn-master")
    parser.add_argument("--data-path", default="/data_a/lsy/code/burst-ann2snn-master/dataset")
    parser.add_argument("--dataset", default="cifar10dvs")
    parser.add_argument("--arch", default="resnet18")
    parser.add_argument("--frame-data", type=int, default=0, help="Use pre-integrated frame caches instead of burst/spikingjelly dataset loaders.")
    parser.add_argument("--speck-input", type=int, default=0, help="Collapse 18-bin voxel inputs to Speck DVS polarity channels and resize to --input-size.")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--input-channels", type=int, default=2, choices=[1, 2])
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--pretrained-ann", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-best", type=int, default=1, help="Save best ANN and evaluated SNN checkpoints.")
    parser.add_argument("--save-dir", default="checkpoints", help="Directory for best checkpoint subdirectories.")
    parser.add_argument("--run-name", default="", help="Optional checkpoint subdirectory name.")

    parser.add_argument("--node-num", type=int, default=9)
    parser.add_argument("--R", type=int, default=20)
    parser.add_argument("--E", type=int, default=1)
    parser.add_argument("--sample-ratio", type=float, default=0.5)
    parser.add_argument("--snn-ratio", type=float, default=0.5)
    parser.add_argument("--partition", choices=["iid", "dir"], default="iid")
    parser.add_argument("--dir-alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--optimizer", default="adamw", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.06)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--augment", type=int, default=0)
    parser.add_argument("--crop-padding", type=int, default=8)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--temporal-shift", type=int, default=2)
    parser.add_argument("--bin-drop-prob", type=float, default=0.1)
    parser.add_argument("--polarity-drop-prob", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--cutmix-alpha", type=float, default=0.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)

    parser.add_argument("--test-T", type=int, default=8)
    parser.add_argument("--calib", default="light", choices=["none", "light"])
    parser.add_argument("--calib-iters", type=int, default=5)
    parser.add_argument("--maxspike", type=int, default=4)
    parser.add_argument("--initialspike", type=int, default=8)
    parser.add_argument("--minspike", type=int, default=1)
    parser.add_argument("--burst-search", type=int, default=1, help="Use sensitivity max-spike search. Threshold search is intentionally not used.")
    parser.add_argument("--searchtime", type=int, default=1)
    parser.add_argument("--maxspike-ratio", type=float, default=1.0)
    parser.add_argument("--search-method", default="pruning")
    parser.add_argument("--metric", default="kl")

    parser.add_argument("--zo-epsilon", type=float, default=0.01)
    parser.add_argument("--zero-min", type=float, default=-0.5)
    parser.add_argument("--zero-max", type=float, default=0.5)
    parser.add_argument("--zero-momentum", type=float, default=0.2)
    parser.add_argument("--eta-snn", type=float, default=0.1)
    parser.add_argument(
        "--executor-direction",
        choices=("guided", "random"),
        default="guided",
        help="Use the SyNAX-guided correction direction or a unit-norm random direction with the same scalar step.",
    )
    parser.add_argument("--feedback-batches", type=int, default=10)
    parser.add_argument(
        "--snn-feedback-warmup-rounds",
        type=int,
        default=0,
        help="Train the ANN clients without SNN executor correction for the first N rounds.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0, help="Limit ANN local batches per epoch for smoke tests. 0 means no limit.")
    parser.add_argument("--eval-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    add_burst_paths(args.burst_root)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_random_seed(args.seed)
    print(f"Running SyNAX-Neuro on {args.device}")
    print(f"Dataset={args.dataset}, arch={args.arch}, burst_search={bool(args.burst_search)}, threshold_search=False")

    data = build_neuro_data(args)
    add_train_augmentation(data, args)
    global_model = build_neuro_model(args.arch, data.num_classes).to(args.device)
    load_checkpoint_if_needed(global_model, args.pretrained_ann, args.device)

    checkpoint_dir = ""
    if args.save_best:
        run_name = args.run_name or f"{args.dataset}_{args.arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
        checkpoint_dir = os.path.join(args.save_dir, run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Saving best checkpoints to: {checkpoint_dir}")

    print("Initial burst ANN2SNN conversion...")
    global_snn = convert_ann_to_burst_snn(global_model, data.calib_loader, args)
    previous_snn = copy.deepcopy(global_snn)
    g_t = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_model.state_dict().items() if v.is_floating_point()}
    G_t = compute_snn_guidance(global_snn, previous_snn)

    ann_count = args.node_num - int(args.node_num * args.snn_ratio)
    device_types = ["ANN" if idx < ann_count else "SNN" for idx in range(args.node_num)]
    best_ann_acc = -1.0
    best_snn_acc = -1.0

    for round_idx in range(args.R):
        round_lr = scheduled_lr(args.lr, round_idx, args.R, args.warmup_rounds)
        print(f"=============== SyNAX-Neuro Round {round_idx + 1}/{args.R} lr={round_lr:.6g} ===============")
        sample_size = max(1, int(args.node_num * args.sample_ratio))
        sampled = random.sample(range(args.node_num), sample_size)
        ann_nodes = [idx for idx in sampled if device_types[idx] == "ANN"]
        snn_nodes = [idx for idx in sampled if device_types[idx] == "SNN"]

        ann_updates = []
        for node_idx in ann_nodes:
            local_model = copy.deepcopy(global_model).to(args.device)
            old_state = copy.deepcopy(local_model.state_dict())
            steps = max(1, args.E * len(data.train_loader[node_idx]))
            gamma = 1.0 / max(steps * round_lr, 1e-12)
            train_synax_ann(local_model, data.train_loader[node_idx], args, g_t, gamma, round_lr, round_idx, node_idx + 1)
            ann_updates.append(tensor_delta(local_model.state_dict(), old_state))
            del local_model
            torch.cuda.empty_cache()

        avg_update = average_updates(ann_updates, global_model.state_dict())

        alphas = []
        if round_idx < args.snn_feedback_warmup_rounds:
            print(
                f"SNN feedback warmup: round={round_idx + 1}/"
                f"{args.snn_feedback_warmup_rounds}, executor correction disabled"
            )
        else:
            for node_idx in snn_nodes:
                alpha = compute_snn_feedback(global_snn, data.train_loader[node_idx], G_t, args)
                if math.isfinite(alpha):
                    alphas.append(alpha)
        avg_alpha = sum(alphas) / len(alphas) if alphas else 0.0
        clamped_alpha = max(args.zero_min, min(args.zero_max, avg_alpha))
        print(f"Aggregated alpha: {avg_alpha:.6f} -> {clamped_alpha:.6f}")

        old_ann_state = copy.deepcopy(global_model.state_dict())
        apply_update(global_model, avg_update, scale=1.0)
        apply_normalized_executor_correction(
            global_model,
            g_t,
            clamped_alpha,
            args.eta_snn,
            direction_mode=args.executor_direction,
            random_seed=args.seed * 1_000_003 + round_idx,
        )

        g_t = tensor_delta(old_ann_state, global_model.state_dict())
        previous_snn = copy.deepcopy(global_snn)
        global_snn = convert_ann_to_burst_snn(global_model, data.calib_loader, args)
        G_t = compute_snn_guidance(global_snn, previous_snn)

        ann_acc = evaluate_ann(global_model, data.test_loader, args.device)
        print(f"Round {round_idx + 1} ANN Acc: {ann_acc:.2f}%")
        if args.save_best and ann_acc > best_ann_acc:
            best_ann_acc = ann_acc
            save_training_checkpoint(
                os.path.join(checkpoint_dir, "best_ann.pt"),
                round_idx=round_idx + 1,
                args=args,
                ann_model=global_model,
                ann_acc=ann_acc,
            )
            print(f"Saved best ANN checkpoint: round={round_idx + 1}, acc={ann_acc:.2f}%")
        if (round_idx + 1) % args.eval_every == 0 or (round_idx + 1) == args.R:
            snn_acc, spikecount = evaluate_snn(global_snn, data.test_loader, args.device)
            print(f"Round {round_idx + 1} SNN Acc: {snn_acc:.2f}% | spikecount={spikecount:.1f}")
            if args.save_best and snn_acc > best_snn_acc:
                best_snn_acc = snn_acc
                save_training_checkpoint(
                    os.path.join(checkpoint_dir, "best_snn.pt"),
                    round_idx=round_idx + 1,
                    args=args,
                    ann_model=global_model,
                    snn_model=global_snn,
                    ann_acc=ann_acc,
                    snn_acc=snn_acc,
                    spikecount=spikecount,
                )
                print(
                    f"Saved best SNN checkpoint: round={round_idx + 1}, "
                    f"snn_acc={snn_acc:.2f}%, ann_acc={ann_acc:.2f}%"
                )

    print("Finished SyNAX-Neuro.")


if __name__ == "__main__":
    main()
