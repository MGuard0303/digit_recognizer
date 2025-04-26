import pandas as pd
import torch


# Convert CSV data to Tensor.
# The shape of output data tensor is (N, H, W).
def csv_to_tensor(dataframe: pd.DataFrame, is_train: bool = True) -> list:
    if is_train is True:
        data = torch.tensor(dataframe.iloc[:, 1:].to_numpy(), dtype=torch.float)
        data = data.reshape(data.size(0), 28, 28)
        label = torch.tensor(dataframe.iloc[:, 1].to_numpy(), dtype=torch.float)

        return [data, label]
    else:
        data = torch.tensor(dataframe.to_numpy(), dtype=torch.float)
        data = data.reshape(data.size(0), 28, 28)

        return [data]


# Use min-max normalization to shrink pixel value to [0, 1].
# The size of inputs and outputs tensors are both (N, H, W).
# Note: Be sure dtype of inputs should be float.
def normalize(inputs: torch.Tensor, norm: str = None) -> torch.Tensor:
    if norm == "global":
        glo_max = torch.max(inputs).item()
        glo_min = torch.min(inputs).item()
        inputs = (inputs - glo_min) / (glo_max - glo_min)

    elif norm == "per-image":
        inputs = torch.flatten(inputs, start_dim=1)
        img_max = torch.max(inputs, dim=1)[0].unsqueeze(1)
        img_min = torch.min(inputs, dim=1)[0].unsqueeze(1)
        inputs = (inputs - img_min) / (img_max - img_min)
        inputs = inputs.reshape(inputs.size(0), 28, 28)

    return inputs
