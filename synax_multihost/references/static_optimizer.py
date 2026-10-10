# Extracted reference functions; do not run as a training script.
from __future__ import annotations
import copy
import math
import torch
from torch import nn


def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, eps=args.opt_eps,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999),
            eps=args.opt_eps, weight_decay=args.weight_decay,
        )
    return optimizer
