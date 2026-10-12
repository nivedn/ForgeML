import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 32, bias=False)

    def forward(self, x):
        output = self.fc1(x)
        return output


if __name__ == "__main__":
    x = torch.zeros(8, 10)
    model = Linear()
    output = model(x)

    # Print the model stats
    print(model)
    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
