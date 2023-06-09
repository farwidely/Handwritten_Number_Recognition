import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


# 程序运行实验使用 GTX 1650 Super 显卡

# MyMNIST1用完整数据集训练20轮，耗时30min，测试精度可达到0.99
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


# MyMNIST2用完整数据集训练20轮，耗时3min24s，测试精度可达到0.99
class MyMNIST2(nn.Module):
    def __init__(self):
        super(MyMNIST2, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=1),
            nn.Flatten(),
            nn.Linear(27 * 27 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# MyMNIST3用完整数据集训练20轮，耗时2min20s，测试精度可达到0.985
class MyMNIST3(nn.Module):
    def __init__(self):
        super(MyMNIST3, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(14 * 14 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 测试模型能否运行
if __name__ == '__main__':
    model = MyMNIST3()
    input = torch.ones((64, 1, 28, 28))
    output = model(input)
    # print(output.shape)
    writer = SummaryWriter("log_models")
    writer.add_graph(model, input)
    writer.close()
