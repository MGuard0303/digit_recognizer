import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import logics
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("./dataset/test.csv")

data_test = utils.csv_to_tensor(dataframe=df, is_training=False)
data_test = utils.normalize(inputs=data_test, normalization="per-image")
data_test.to(device)

# Wrap data to Dataset and DataLoader
ds_test = TensorDataset(data_test)
dl_test = DataLoader(ds_test, batch_size=256, shuffle=False)

# Prediction setup
model = torch.load("./expt/20250501/113400.pt")
model.to(device)
