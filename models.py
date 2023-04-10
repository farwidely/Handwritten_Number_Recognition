import torch
from torch import nn


# MyMNIST1用完整数据集训练，测试精度可达到0.99
# 若只用十分之一的数据训练，测试精度可达到0.96
class MyMNIST1(nn.Module):
    def __init__(self):
        super(MyMNIST1, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
            nn.Flatten(),
            nn.Linear(26 * 26 * 64, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# MyMNIST2用完整数据集训练，测试精度可达到0.985
# 若只用十分之一的数据训练，测试精度可达到0.95
class MyMNIST2(nn.Module):
    def __init__(self):
        super(MyMNIST2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
            nn.Flatten(),
            nn.Linear(27 * 27 * 16, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 测试模型能否运行
if __name__ == '__main__':
    model = MyMNIST1()
    input = torch.ones((64, 1, 28, 28))
    output = model(input)
    print(output.shape)
