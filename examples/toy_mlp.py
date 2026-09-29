import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 32)

    def forward(self, x):
        x1 = self.fc1(x)
        output = F.relu(x1)
        return output


if __name__ == "__main__":
    x = torch.zeros(8, 10)
    model = Mlp()
    output = model(x)

    # Print the model stats
    print(model)
    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
