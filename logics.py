import copy

import torch

from torch.utils.data import DataLoader


def train_step(model: torch.nn.Module, img: torch.Tensor, label: torch.Tensor) -> tuple:
    model.train()
    model.optimizer.zero_grad()

    # Forward
    # TODO
    