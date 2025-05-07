import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import dlmodel
import logics
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("./dataset/test.csv")

data_test = utils.csv_to_tensor(dataframe=df, is_training=False)
data_test = utils.normalize(inputs=data_test, normalization="per-image")
data_test = data_test.to(device)

# Wrap data to Dataset and DataLoader
ds_test = TensorDataset(data_test)
dl_test = DataLoader(ds_test, batch_size=256, shuffle=False)

# Prediction setup
params = torch.load("expt/20250506/093248.pt", weights_only=True)
model = dlmodel.LeNet()
model.load_state_dict(params["param"])
model.epoch_loss_trn = params["train_loss"]
model.epoch_loss_vld = params["valid_loss"]
model.to(device)
prediction = logics.predict(model=model, loader_predict=dl_test, is_return=True)
