import torch
from torch import nn

class MyMNIST(nn.Module):
    def __init__(self):
        super(MyMNIST, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Flatten(),
            nn.Linear(28*28*64, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = MyMNIST()
    input = torch.ones((64, 1, 28, 28))
    output = model(input)
    print(output.shape)