import torch.nn.functional as F
from torch import nn


class TwoOutputs(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 4)

    def forward(self, x):
        y = self.fc(x)
        return F.relu(y), y
