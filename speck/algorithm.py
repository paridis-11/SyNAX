"""Math and round policies aligned with static/SyNAX_normalized_snn.py."""

import math

import torch


def unit_direction(values, minimum_norm=0.0):
    floating = {key: value for key, value in values.items() if value.is_floating_point()}
    squared = sum(float(value.detach().double().square().sum()) for value in floating.values())
    if not math.isfinite(squared):
        raise ValueError("Non-finite guidance")
    norm = math.sqrt(squared)
    return {key: value / norm if norm > minimum_norm else torch.zeros_like(value)
            for key, value in floating.items()}


def snn_guidance(previous, current):
    difference = {key: previous[key].detach().float() - value.detach().float()
                  for key, value in current.items()
                  if value.is_floating_point() and key in previous
                  and value.shape == previous[key].shape and not key.endswith(".v_mem")}
    return difference


def global_state_update(state, parameter_names, average_delta, guidance, alpha,
                        eta_ann, eta_snn):
    valid = {name: guidance[name] for name in parameter_names
             if name in guidance and guidance[name].is_floating_point()}
    direction = unit_direction(valid, minimum_norm=1e-12)
    output = {}
    for name, value in state.items():
        updated = value.clone()
        if name in average_delta:
            updated = updated + eta_ann * average_delta[name].to(updated)
        if value.is_floating_point() and name in direction:
            updated = updated - eta_snn * alpha * direction[name].to(updated)
        output[name] = updated.to(value.dtype)
    return output


def scheduled_lr(initial_lr, round_idx, rounds):
    return initial_lr * 0.5 * (1 + math.cos(math.pi * round_idx / rounds))


def sample_clients(roster, ratio, rng):
    indices = rng.sample(range(len(roster)), max(1, int(len(roster) * ratio)))
    if not any(roster[index]["role"] == "ann" for index in indices):
        ann_indices = [index for index, member in enumerate(roster) if member["role"] == "ann"]
        indices[rng.randrange(len(indices))] = rng.choice(ann_indices)
    return [roster[index]["id"] for index in indices]


def mixed_gradient(gradient, guidance, beta, gamma):
    return beta * gradient + (1 - beta) * gamma * guidance.to(gradient)
