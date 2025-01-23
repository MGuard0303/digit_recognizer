import copy

import torch

from torch.utils.data import DataLoader


def train_step(model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor) -> tuple:
    model.train()
    model.optimizer.zero_grad()

    # Forward
    pred = model(img=image)
    loss = model.loss_function(pred, label)

    # Backward
    loss.backward()
    model.optimizer.step()

    return loss.item(), pred


@torch.no_grad()
def valid_step(model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor) -> tuple:
    model.eval()

    pred = model(img=image)
    loss = model.loss_function(pred, label)

    return loss.item(), pred


def train(model: torch.nn.Module, train_loader: DataLoader, valid_loader: DataLoader, epochs: int,
          valid_per_epochs: int, is_return: bool = False) -> tuple:
    # TODO: Early-stopping.

    for epoch in range(1, epochs + 1):
        trn_loss = 0.0

        # Training step.
        num_trn_steps = len(train_loader)

        for _, (img, lbl) in enumerate(train_loader):
            # TODO: Check shape of label tensor.
            lbl = lbl.squeeze(1)  # The shape of NLLLoss is (N, C) or (C).
            lbl = lbl.type(torch.long)
            batch_trn_loss, _ = train_step(model=model, image=img, label=lbl)
            trn_loss += batch_trn_loss

        avg_trn_loss = trn_loss / num_trn_steps
        print(f"Epoch {epoch:02}")
        print(f"| Average Training Loss: {avg_trn_loss:.3f} |")

        # Validate model at specified epoch.
        if epoch % valid_per_epochs == 0:
            print(f"Validating at Epoch {epoch:02}")
            vld_loss = 0.0
            num_vld_steps = len(valid_loader)

            for _, (img, lbl) in enumerate(valid_loader):
                lbl = lbl.sequeeze(1)  # TODO: Check.
                lbl = lbl.type(torch.long)
                batch_vld_loss, _ = valid_step(model=model, image=img, label=lbl)
                vld_loss += batch_vld_loss

            avg_vld_loss = vld_loss / num_vld_steps
            print(f"| Average Validation Loss: {avg_vld_loss:.3f} |")

            # TODO: Select best model and early-stopping

    print("Training Complete.")

    if is_return:
        # TODO: Return value.
