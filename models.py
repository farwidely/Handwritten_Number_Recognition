import torch
import torchvision
from torch import nn


class MyMNIST(nn.Module):
    def __init__(self):
        super(MyMNIST, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
            nn.Flatten(),
            nn.Linear(26*26*32, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = MyMNIST()
    input = torch.ones((64, 1, 28, 28))
    output = model(input)
    print(output.shape)