import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import dlmodel
import logics
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("./dataset/train.csv")

data, label = utils.csv_to_tensor(df)
data = utils.normalize(data, normalization="per-image")

random_index = torch.randperm(data.size(0))
num_valid = int(data.size(0) * 0.3)
index_valid = random_index[:num_valid]
index_train = random_index[num_valid:]

data_train = data[index_train].to(device)
data_valid = data[index_valid].to(device)
label_train = label[index_train].to(device)
label_valid = label[index_valid].to(device)

# Wrap data to Dataset and DataLoader
ds_train = TensorDataset(data_train, label_train)
ds_valid = TensorDataset(data_valid, label_valid)

dl_train = DataLoader(ds_train, batch_size=256, shuffle=True)
dl_valid = DataLoader(ds_valid, batch_size=256, shuffle=True)

model = dlmodel.LeNet()
model.optimizer = torch.optim.Adam(model.parameters())
model.to(device)

# Training setup
epoch = 10
model_final = logics.train(model=model, loader_train=dl_train, loader_valid=dl_valid, epochs=epoch, is_return=True)
