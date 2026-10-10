import argparse
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .algorithm import mixed_gradient, unit_direction


DEFAULTS = dict(E=5, lr=0.001, optimizer="adamw", momentum=0.0, weight_decay=0.001, opt_eps=1e-8,
                zero_momentum=1.0, zo_epsilon=0.01, zero_min=-0.5, zero_max=0.5,
                time_steps=40, batch_size=32, workers=0, feedback_batches=10,
                max_train_batches=0, spike_threshold=1.0, min_v_mem=-1.0,
                add_spiking_output=1, seed=42, eta_ann=1.0, eta_snn=0.1,
                hw_drain_seconds=0.2, sample_ratio=1.0, drop_last=True)


def validate_options(options):
    for name in ("E", "time_steps", "batch_size", "feedback_batches"):
        if not isinstance(options[name], int) or options[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("workers", "max_train_batches"):
        if type(options[name]) is not int or options[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if options["optimizer"] not in {"sgd", "adam", "adamw"}:
        raise ValueError("Unsupported ANN optimizer")
    for name in ("lr", "zo_epsilon", "spike_threshold", "hw_drain_seconds", "opt_eps"):
        if not math.isfinite(options[name]) or options[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not 0 <= options["zero_momentum"] <= 1:
        raise ValueError("zero_momentum must be in [0, 1]")
    if not 0 < options["sample_ratio"] <= 1:
        raise ValueError("sample_ratio must be in (0, 1]")
    if type(options["drop_last"]) is not bool:
        raise ValueError("drop_last must be a boolean")
    for name in ("eta_ann", "eta_snn", "zero_min", "zero_max", "min_v_mem",
                 "momentum", "weight_decay"):
        if not math.isfinite(options[name]):
            raise ValueError(f"{name} must be finite")
    if options["zero_min"] > options["zero_max"]:
        raise ValueError("zero_min must not exceed zero_max")


def event_module():
    return importlib.import_module("synax_speck_event")


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if not key.endswith(".v_mem")}


def parameter_delta(new, old, names):
    return {key: new[key].float() - old[key].float() for key in names}


def normalized(delta):
    return unit_direction(delta)


def new_model(spec, device="cpu"):
    if spec["backend"] == "toy":
        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
    options = argparse.Namespace(**spec["options"], device=torch.device(device))
    return event_module().build_model(options, spec["num_classes"]).to(device)


def convert(model, spec):
    if spec["backend"] == "toy":
        return copy.deepcopy(model).cpu().eval()
    options = argparse.Namespace(**spec["options"], device=torch.device("cpu"))
    return event_module().convert_ann_to_snn(model, options).cpu().eval()


def initial_state(backend, checkpoint, overrides):
    if set(overrides) - set(DEFAULTS):
        raise ValueError(f"Unknown hyperparameters: {sorted(set(overrides) - set(DEFAULTS))}")
    options = DEFAULTS.copy()
    torch.manual_seed(overrides.get("seed", 42))
    if backend == "event":
        if not checkpoint:
            raise ValueError("Event coordinator requires an existing --checkpoint")
        # A locally selected checkpoint, not an untrusted network object.
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        archived = saved.get("args", {})
        options.update({key: value for key, value in archived.items() if key in options})
        options.update({key: archived[key] for key in ("arch", "conv_bias", "input_shape", "dataset")
                        if key in archived})
        if options.get("dataset") not in {"nmnist", "dvsgesture"}:
            raise ValueError("Hardware adapter supports nmnist and dvsgesture")
        classes, sensor = event_module().dataset_spec(argparse.Namespace(**options))
        options.setdefault("input_shape", (sensor[2], sensor[1], sensor[0]))
        options.setdefault("arch", "large64")
        options.setdefault("conv_bias", 1)
    else:
        saved, classes = None, 2
        options["drop_last"] = False
    options.update(overrides)
    validate_options(options)
    spec = dict(backend=backend, options=options, num_classes=classes,
                algorithm_version="synax-static-v2")
    model = new_model(spec)
    if saved:
        model.load_state_dict(saved["ann_model"], strict=True)
    snn = convert(model, spec)
    if saved and "snn_model" in saved:
        incompatible = snn.load_state_dict({key: value for key, value in saved["snn_model"].items()
                                           if not key.endswith(".v_mem")}, strict=False)
        if incompatible.unexpected_keys or any(not key.endswith(".v_mem")
                                              for key in incompatible.missing_keys):
            raise ValueError(f"Checkpoint SNN structure mismatch: {incompatible}")
    return spec, model, snn


class RawFrames(Dataset):
    def __init__(self, raw, spec):
        self.raw, self.spec = raw, spec

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        events, label = self.raw[index]
        helper = event_module()
        if self.spec["options"]["dataset"] == "nmnist":
            options = self.spec["options"]
            size = options["input_shape"]
            frames = helper.ToFrame(sensor_size=(size[2], size[1], size[0]),
                                    n_time_bins=options["time_steps"])(events)
        else:
            size = self.spec["options"]["input_shape"]
            frames = helper.EventNPZToFrameDataset._events_to_frames_static(
                events, self.spec["options"]["time_steps"], (size[2], size[1], size[0]))
        return torch.as_tensor(frames, dtype=torch.float32), int(label)


def local_data(spec, root, shard, shards, indices_file=None):
    if spec["backend"] == "toy":
        generator = torch.Generator().manual_seed(spec["options"]["seed"])
        x = torch.randn(96, 4, generator=generator)
        dataset = TensorDataset(x, (x[:, 0] + x[:, 1] > 0).long())
        raw = None
    else:
        helper = event_module()
        if spec["options"]["dataset"] == "nmnist":
            raw = helper.NMNIST(save_to=root, train=True)
            # Filesystem traversal order can differ between physical hosts.
            base = Path(root).resolve()
            order = sorted(range(len(raw)), key=lambda index: str(
                Path(raw.data[index]).resolve().relative_to(base)))
            raw = Subset(raw, order)
        else:
            raw = helper.SplitEventNPZRawDataset(root, "train")
        dataset = RawFrames(raw, spec)
    if indices_file:
        indices = json.loads(Path(indices_file).read_text())
        if (not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < len(dataset)
                                                for i in indices)
                or len(indices) != len(set(indices))):
            raise ValueError("Indices must be unique valid training-set indices")
    else:
        indices = list(range(len(dataset)))
        random.Random(spec["options"]["seed"]).shuffle(indices)
        indices = indices[shard::shards]
    if not indices:
        raise ValueError("Empty local training shard")
    return dataset, raw, indices


def make_optimizer(model, options, lr):
    if options["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=options["momentum"],
                               weight_decay=options["weight_decay"])
    if options["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, eps=options["opt_eps"],
                                weight_decay=options["weight_decay"])
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=options["opt_eps"],
                             weight_decay=options["weight_decay"])


def ann_result(job, device, root, indices_file):
    spec, options = job["spec"], job["spec"]["options"]
    dataset, _, indices = local_data(spec, root, job["shard"], job["shards"], indices_file)
    loader = DataLoader(Subset(dataset, indices), batch_size=options["batch_size"],
                        shuffle=True, drop_last=options["drop_last"], num_workers=options["workers"])
    if not len(loader):
        raise ValueError("Shard has no full minibatch; lower batch_size or explicitly set drop_last=false")
    model = new_model(spec, device)
    model.load_state_dict(job["ann_state"], strict=True)
    lr = job.get("round_lr", options["lr"])
    args = argparse.Namespace(**dict(options, lr=lr), device=torch.device(device))
    optimizer = make_optimizer(model, options, lr)
    # The formal static runner uses K = E * len(loader), including batch scheduling.
    gamma = 1.0 / (options["E"] * len(loader) * lr) if lr else 1.0
    model.train()
    for _ in range(options["E"]):
        for batch_idx, (x, y) in enumerate(loader):
            if options["max_train_batches"] and batch_idx >= options["max_train_batches"]:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = (event_module().temporal_logits(model, x, args.device)
                      if spec["backend"] == "event" else model(x.to(device)))
            nn.functional.cross_entropy(logits, y.long().to(device)).backward()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and name in job["ann_guidance"]:
                        parameter.grad.copy_(mixed_gradient(parameter.grad, job["ann_guidance"][name],
                                                             options["zero_momentum"], gamma))
            optimizer.step()
    delta = parameter_delta(cpu_state(model), job["ann_state"], job["ann_guidance"])
    return dict(delta=delta, samples=len(indices))


def perturbed_snn(job, sign):
    model = convert(new_model(job["spec"]), job["spec"])
    model.load_state_dict(job["snn_state"], strict=False)
    with torch.no_grad():
        direction = unit_direction(job["snn_guidance"])
        for name, parameter in model.named_parameters():
            if name in direction:
                parameter.add_(direction[name].to(parameter),
                               alpha=sign * job["spec"]["options"]["zo_epsilon"])
    return model.eval()


def hardware_weight_digest(config):
    arrays = [{"weights": layer.weights, "biases": getattr(layer, "biases", []),
               "threshold_high": getattr(layer, "threshold_high", None),
               "threshold_low": getattr(layer, "threshold_low", None)}
              for layer in config.cnn_layers]
    return hashlib.sha256(json.dumps(arrays, sort_keys=True).encode()).hexdigest()


def replay_events(net, events, drain_seconds):
    from sinabs.backend.dynapcnn.io import disable_timestamps, enable_timestamps, reset_timestamps

    net.samna_output_buffer.get_events()
    reset_timestamps(net.device)
    enable_timestamps(net.device)
    output = []
    try:
        duration = max((event.timestamp for event in events), default=0) / 1e6
        start = time.monotonic()
        net.samna_input_buffer.write(events)
        # Wait for the last scheduled input, not merely a temporary gap in output.
        deadline = start + duration + drain_seconds
        while time.monotonic() < deadline:
            output.extend(net.samna_output_buffer.get_events())
            time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        output.extend(net.samna_output_buffer.get_events())
        return output
    finally:
        disable_timestamps(net.device)


def stop_hardware(net):
    import samna
    try:
        for name in ("device_input_graph", "device_output_graph", "samna_input_graph"):
            graph = getattr(net, name, None)
            if graph is not None:
                graph.stop()
    finally:
        samna.device.close_device(net.samna_device)


@torch.no_grad()
def executor_result(job, mode, root, indices_file, speck_device):
    spec, options = job["spec"], job["spec"]["options"]
    dataset, raw, indices = local_data(spec, root, job["shard"], job["shards"], indices_file)
    generator = random.Random(options["seed"] + 1009 * job["round"] + job["shard"])
    available = (len(indices) // options["batch_size"] * options["batch_size"]
                 if options["drop_last"] else len(indices))
    if not available:
        raise ValueError("Executor shard has no full minibatch; lower batch_size or set drop_last=false")
    selected = generator.sample(indices, min(available, options["batch_size"] * options["feedback_batches"]))
    if all(not torch.count_nonzero(value).item() for value in job["snn_guidance"].values()):
        return dict(alpha=0.0, samples=len(selected), skipped="zero guidance", mode=mode)
    plus, minus = perturbed_snn(job, 1), perturbed_snn(job, -1)
    if mode == "speck":
        if spec["backend"] != "event":
            raise ValueError("Speck requires the event backend")
        import samna
        from sinabs.backend.dynapcnn import DynapcnnNetwork
        helper = event_module()
        base = perturbed_snn(job, 0)
        def dyn(model):
            return DynapcnnNetwork(snn=model, input_shape=tuple(options["input_shape"]),
                                  discretize=True, dvs_input=False)
        base_dyn = dyn(base)
        base_config = base_dyn.make_config(device=speck_device, monitor_layers=[-1])
        ordering = base_dyn.chip_layers_ordering
        configs = [dyn(model).make_config(device=speck_device,
                                          chip_layers_ordering=ordering, monitor_layers=[-1])
                   for model in (plus, minus)]
        fingerprints = [hardware_weight_digest(config) for config in configs]
        if fingerprints[0] == fingerprints[1]:
            raise ValueError("Both probes quantize to identical Speck parameters; increase zo_epsilon")
        # Cache raw samples once: both probes replay exactly the same event streams.
        samples = [raw[index] for index in selected]
        net = base_dyn.to(device=speck_device, chip_layers_ordering=ordering, monitor_layers=[-1])
        losses = []
        empty_outputs = []
        try:
            for config in configs:
                # Sinabs reset_states() reapplies samna_config; keep it on the probe.
                net.samna_config = config
                net.samna_device.get_model().apply_configuration(config)
                total_loss = 0.0
                empty = 0
                for events, label in samples:
                    net.reset_states()
                    stream = helper.make_samna_events(events, samna.speck2f.event.Spike)
                    for spike in stream:
                        spike.layer = ordering[0]
                    output = replay_events(net, stream, options["hw_drain_seconds"])
                    counts = torch.zeros(spec["num_classes"])
                    for event in output:
                        feature = int(event.feature)
                        if 0 <= feature < len(counts):
                            counts[feature] += 1
                    empty += int(counts.sum() == 0)
                    total_loss += float(nn.functional.cross_entropy(
                        counts.unsqueeze(0), torch.tensor([int(label)])))
                losses.append(total_loss / len(samples))
                empty_outputs.append(empty)
        finally:
            try:
                net.samna_config = base_config
                net.samna_device.get_model().apply_configuration(base_config)
            finally:
                stop_hardware(net)
    else:
        samples = [dataset[index] for index in selected]
        losses, fingerprints, empty_outputs = [], [], []
        for model in (plus, minus):
            total_loss = 0.0
            for x, label in samples:
                if spec["backend"] == "event":
                    event_module().reset_snn(model)
                    logits = event_module().temporal_logits(model, x.unsqueeze(0), torch.device("cpu"))
                else:
                    logits = model(x.unsqueeze(0))
                total_loss += float(nn.functional.cross_entropy(logits, torch.tensor([int(label)])))
            losses.append(total_loss / len(samples))
    alpha = (losses[0] - losses[1]) / (2 * options["zo_epsilon"])
    if not math.isfinite(alpha):
        raise ValueError("Non-finite executor feedback")
    return dict(alpha=alpha, loss_plus=losses[0], loss_minus=losses[1],
                samples=len(selected), mode=mode, probe_weight_hashes=fingerprints,
                empty_output_samples=empty_outputs)
