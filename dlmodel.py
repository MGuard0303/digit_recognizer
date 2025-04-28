import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.epoch_loss_trn = []
        self.epoch_loss_vld = []

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=(2, 2)),  # (N, C, H, W)
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (N, C, H, W)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),  # (N, C, H, W)
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (N, C, H, W)
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=10),
            nn.LogSoftmax(dim=1),
        )

        self.loss_function = nn.NLLLoss()

    def forward(self, image: torch.Tensor):
        image = self.backbone(image)

        return image
        