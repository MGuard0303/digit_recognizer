import pandas as pd
import torch


def csv_to_tensor(dataframe: pd.DataFrame, is_train: bool = True) -> torch.Tensor:
    if is_train is True:
        tensor = torch.tensor(dataframe.iloc[:, 1:].to_numpy(), dtype=torch.float)
    else:
        tensor = torch.tensor(dataframe.to_numpy(), dtype=torch.float)

    return tensor


# Use min-max normalization to shrink pixel value to [0, 1].
# Note: Be sure dtype of inputs should be float.
def normalize(inputs: torch.Tensor, is_global: bool = False) -> torch.Tensor:
    if is_global is True:
        glo_max = torch.max(inputs).item()
        glo_min = torch.min(inputs).item()
        inputs.apply_(lambda x: (x - glo_min) / (glo_max - glo_min))
    else:
        loc_max, _ = torch.max(inputs, dim=1)
        loc_min, _ = torch.min(inputs, dim=1)

        for batch_data, batch_max, batch_min in zip(inputs, loc_max, loc_min):
            batch_data.apply_(lambda x: (x - batch_min.item()) / (batch_max.item() - batch_min.item()))

    return inputs
