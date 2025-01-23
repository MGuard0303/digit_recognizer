import torch
from torch import nn


class ConvModel(nn.Module):
    def __init__(self, adam_lr: int = 0.001, adam_betas: tuple[float, float] = (0.9, 0.999),
                 adam_weight_decay: int = 0):
        super().__init__()

        self.identity = nn.Sequential()

        self.conv_unit = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5)),
            nn.BatchNorm2d(num_features=32),
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

        self.relu = nn.ReLU()
        self.drop = nn.Dropout()

        self.output_layer = nn.LogSoftmax(dim=1)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=adam_lr,
            betas=adam_betas,
            weight_decay=adam_weight_decay,
        )

        self.loss_function = nn.NLLLoss()

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Convolution Unit
        img = self.conv_unit(img)

        img = nn.Flatten(start_dim=2)(img)

        # First FC Unit
        img_identity = self.identity(img)
        img = self.fc_layers["fc1-1"](img)
        img = self.fc_layers["fc1-2"](img)
        img = nn.BatchNorm1d(num_features=32)(img)
        img = img + img_identity
        img = self.relu(img)

        img = self.fc_layers["fc1-3"](img)
        img = self.relu(img)
        img = self.drop(img)
        img = self.fc_layers["fc1-4"](img)
        img = self.relu(img)
        img = self.drop(img)

        img = nn.Flatten()(img)

        # Second FC Unit
        img = self.fc_layers["fc2-1"](img)
        img = nn.BatchNorm1d(num_features=256)(img)
        img = self.relu(img)
        img = self.drop(img)
        img = self.fc_layers["fc2-2"](img)
        img = self.relu(img)
        img = self.drop(img)
        img = self.fc_layers["fc2-3"](img)
        img = nn.BatchNorm1d(num_features=64)(img)
        img = self.relu(img)
        img = self.drop(img)
        img = self.fc_layers["fc2-4"](img)
        img = self.relu(img)
        img = self.drop(img)
        img = self.fc_layers["fc2-5"](img)

        img = self.output_layer(img)

        return img
