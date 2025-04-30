import torch

from torch.utils.data import DataLoader


def train_process(model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor) -> tuple:
    model.train()  # Set to the training mode.
    model.optimizer.zero_grad()  # Clear gradient everytime.

    # Forward propagation
    pred = model(image=image)
    loss = model.loss_function(pred, label)

    # Backward propagation
    loss.backward()
    model.optimizer.step()  # Update the parameters of the model.

    return loss.item(), pred  # loss is a one-element tensor, so it can use .item() method.


@torch.no_grad()  # This decorator makes following function not calculate gradient.
def valid_process(model: torch.nn.Module, image: torch.Tensor, label: torch.Tensor) -> tuple:
    model.eval()  # Set to the evaluation mode.

    pred = model(image=image)
    loss = model.loss_function(pred, label)

    return loss.item(), pred


@torch.no_grad()
def pred_process(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    model.eval()  # Set to the evaluation mode.

    pred = model(image=image)

    return pred


def train(model: torch.nn.Module, loader_train: DataLoader, loader_valid: DataLoader, epochs: int,
          is_return: bool = False) -> torch.nn.Module:
    print("Start training...")
    steps_trn = len(loader_train)
    steps_vld = len(loader_valid)

    for epoch in range(1, epochs + 1):
        print(f"Training: Epoch {epoch:02}")
        epoch_loss_trn = 0.0

        # Training step.
        for _, (img, lbl) in enumerate(loader_train):
            # noinspection PyTypeChecker
            batch_loss_trn, _ = train_process(model=model, image=img, label=lbl)
            epoch_loss_trn += batch_loss_trn

        avg_epoch_loss_trn = epoch_loss_trn / steps_trn
        model.epoch_loss_trn.append(avg_epoch_loss_trn)

        print(f"| Average Training Loss: {avg_epoch_loss_trn:.3f} |")
        print()

        # Validation.
        print(f"Validating at Epoch {epoch:02}")
        epoch_loss_vld = 0.0

        for _, (img, lbl) in enumerate(loader_valid):
            # noinspection PyTypeChecker
            batch_loss_vld, _ = valid_process(model=model, image=img, label=lbl)
            epoch_loss_vld += batch_loss_vld

        avg_epoch_loss_vld = epoch_loss_vld / steps_vld
        model.epoch_loss_vld.append(avg_epoch_loss_vld)

        print(f"| Average Validation Loss: {avg_epoch_loss_vld:.3f} |")
        print()

    print(" Training complete.")

    if is_return:
        return model


@torch.no_grad()
def predict(model: torch.nn.Module, pred_loader: DataLoader, is_return: bool = False) -> torch.Tensor:
    model.eval()
    preds = []

    for _, img in enumerate(pred_loader):
        # noinspection PyTypeChecker
        batch_pred = pred_process(model=model, image=img)
        batch_pred = torch.exp(batch_pred)
        batch_pred = torch.max(batch_pred, dim=1)[1]
        preds.append(batch_pred)

    pred = torch.cat(preds)

    if is_return:
        return pred
