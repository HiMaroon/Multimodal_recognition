import torch
import torch.nn as nn


class MLPFusionHead(nn.Module):

    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.5
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
