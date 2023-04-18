from torch.utils.data import Subset, DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from models import *

# 准备数据集, 对数据集进行归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# 训练集
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# 测试集
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# 裁剪数据集
train_data = Subset(train_dataset, indices=range(0, 6000))
test_data = Subset(test_dataset, indices=range(0, 1000))

# for i in range(1000):
#     print(train_data[i])

# 查看数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为: {train_data_size}")
print(f"测试数据集的长度为: {test_data_size}")

# 设置Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

model = MyMNIST3()

for data in train_dataloader:
    imgs, targets = data
    outputs = model(imgs)
    print(outputs)

