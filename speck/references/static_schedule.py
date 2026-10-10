# Extracted reference functions; do not run as a training script.
from __future__ import annotations
import copy
import math
import torch
from torch import nn


def LR_scheduler(lr_initial, current_epoch, total_epochs):

    lr_new = lr_initial * 0.5 * (1 + math.cos(math.pi * current_epoch / total_epochs))

    return lr_new
