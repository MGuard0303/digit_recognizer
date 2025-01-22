import torch
from torch import nn


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.identity = nn.Sequential()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3))
        )

        self.fc_layers = nn.ModuleDict({
            "fc1-1": nn.Linear(in_features=64, out_features=128),
            "fc1-2": nn.Linear(in_features=128, out_features=64),
            "fc1-3": nn.Linear(in_features=64, out_features=32),
            "fc1-4": nn.Linear(in_features=32, out_features=16),
            "fc2-1": nn.Linear(in_features=512, out_features=256),
            "fc2-2": nn.Linear(in_features=256, out_features=128),
            "fc2-3": nn.Linear(in_features=128, out_features=64),
            "fc2-4": nn.Linear(in_features=64, out_features=32),
            "fc2-5": nn.Linear(in_features=32, out_features=10),
        })

        self.bn_layers = nn.ModuleDict({
            "bn1": nn.BatchNorm1d(num_features=32),
        })

        self.dropout = nn.Dropout()

        self.output_layer = nn.LogSoftmax(dim=2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # First part: Convolution Unit
        img = self.conv_unit(img)
