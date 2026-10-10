# Extracted reference functions; do not run as a training script.
from __future__ import annotations
import copy
import math
import torch
from torch import nn


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


def scheduled_lr(initial_lr: float, round_idx: int, rounds: int, warmup_rounds: int) -> float:
    if warmup_rounds > 0 and round_idx < warmup_rounds:
        return initial_lr * float(round_idx + 1) / warmup_rounds
    progress = (round_idx - warmup_rounds) / max(1, rounds - warmup_rounds)
    return initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_snn_guidance(current_snn: nn.Module, previous_snn: nn.Module) -> Dict[str, torch.Tensor]:
    current_state = current_snn.state_dict()
    previous_state = previous_snn.state_dict()
    guidance = {}
    flat = []
    for key, value in current_state.items():
        if value.is_floating_point() and key in previous_state:
            diff = previous_state[key].detach().float() - value.detach().float()
            guidance[key] = diff
            flat.append(diff.flatten())
    if not flat:
        return guidance
    norm = torch.norm(torch.cat(flat))
    if norm > 0:
        guidance = {key: value / norm for key, value in guidance.items()}
    return guidance
