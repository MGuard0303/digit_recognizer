import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import dlmodel
import logics
import utils


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("./dataset/train.csv")

data_train, label_train = utils.csv_to_tensor(df)
data_train = utils.normalize(data_train, normalization="per-image")


model = dlmodel.LeNet()
