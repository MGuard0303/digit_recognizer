import matplotlib.pyplot as plt
import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.epoch_loss_trn = []
        self.epoch_loss_vld = []

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=(2, 2)),  # (N, C, H, W)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (N, C, H, W)
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),  # (N, C, H, W)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),  # (N, C, H, W)
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=10),
            nn.LogSoftmax(dim=1),
        )

        self.loss_function = nn.NLLLoss()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.backbone(image)

        return image

    def plot_loss(self,) -> None:
        fig, ax = plt.subplots()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

        ax.plot(self.epoch_loss_trn, label="train loss")
        ax.plot(self.epoch_loss_vld, label="valid loss")

        ax.legend()

        plt.show()
