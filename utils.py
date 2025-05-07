from pathlib import Path

import pandas as pd
import torch


# Convert CSV data to Tensor.
# The shape of output data tensor is (N, C, H, W).
def csv_to_tensor(dataframe: pd.DataFrame, is_training: bool = True) -> list | torch.Tensor:
    if is_training is True:
        data = torch.tensor(dataframe.iloc[:, 1:].to_numpy(), dtype=torch.float)
        data = data.reshape(data.size(0), 28, 28)
        data = torch.unsqueeze(data, dim=1)
        label = torch.tensor(dataframe.iloc[:, 0].to_numpy(), dtype=torch.long)

        return [data, label]
    else:
        data = torch.tensor(dataframe.to_numpy(), dtype=torch.float)
        data = data.reshape(data.size(0), 28, 28)
        data = torch.unsqueeze(data, dim=1)

        return data


# Use min-max normalization to shrink pixel value to [0, 1].
# The size of inputs and outputs tensors are both (N, C, H, W).
# Note: Be sure dtype of inputs should be float.
def normalize(inputs: torch.Tensor, normalization: str = None) -> torch.Tensor:
    if normalization == "global":
        glo_max = torch.max(inputs).item()
        glo_min = torch.min(inputs).item()
        inputs = (inputs - glo_min) / (glo_max - glo_min)

    elif normalization == "per-image":
        inputs = torch.flatten(inputs, start_dim=1)
        img_max = torch.max(inputs, dim=1)[0].unsqueeze(1)
        img_min = torch.min(inputs, dim=1)[0].unsqueeze(1)
        inputs = (inputs - img_min) / (img_max - img_min)
        inputs = inputs.reshape(inputs.size(0), 28, 28)
        inputs = torch.unsqueeze(inputs, dim=1)

    return inputs


def save_param(model: torch.nn.Module, directory: str, filename: str) -> None:
    directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True)

    param_path = Path(f"{directory}/{filename}")
    params = {
        "param": model.state_dict(),
        "train_loss": model.epoch_loss_trn,
        "valid_loss": model.epoch_loss_vld
    }
    torch.save(params, param_path)


def save_prediction(prediction: torch.Tensor, directory: str, filename: str) -> None:
    directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True)

    path = Path(f"{directory}/{filename}")

    if prediction.is_cuda:
        prediction = prediction.cpu()

    prediction = prediction.numpy()

    with open(path, "a",) as f:
        f.write("ImageID, Label\n")

        for i in range(prediction.size):
            f.write(f"{i + 1}, {int(prediction[i])}\n")
