import torch
from torch import nn


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),  # (B, 32, 64, 3000)
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)),  # (B, 64, 32, 3000)
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=(3, 5), stride=(2, 5), padding=(1, 2)),  # (B, 64, 16, 600)
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),  # (B, 64, 8, 600)
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=(8, 1)),  # (B, 128, 1, 600)
        )

        self.tcn = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, dilation=d, padding=d),
            )
            for d in (1, 2, 4, 8, 16, 32, 64, 128)
        ])

    def forward(self, x):
        x = self.cnn(x).squeeze(2) # (B, 128, 600)
        for block in self.tcn:
            x = x + block(x)
        return x 

    
class RegressionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(128, 4, kernel_size=1)  # (B, 4, 600)

    def forward(self, x):
        return self.conv(x)  # (B, 4, 600)


class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.ReLU(),
            )
        

        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.conv(x)  # (B, 128, 600)
        x = torch.cat([x.mean(-1), x.amax(-1)], dim=1)
        return self.mlp(x)  # (B, 1), raw logit


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.regression_head = RegressionHead()
        self.classification_head = ClassificationHead()

    def forward(self, x):
        features = self.backbone(x)
        signals = self.regression_head(features)
        logit = self.classification_head(features)
        return signals, logit