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
            "layer1": nn.Linear(in_features=64, out_features=128),
            "layer2": nn.Linear(in_features=128, out_features=64),
            "layer3": nn.Linear(in_features=64, out_features=32),
            "layer4": nn.Linear(in_features=32, out_features=10),
        })

        self.bn_layers = nn.ModuleDict({
            "bn1": nn.BatchNorm1d(num_features=32),
        })

        self.dropout = nn.Dropout()

        self.output_layer = nn.LogSoftmax(dim=2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # First part: Convolution Unit
        img = self.conv_unit(img)
