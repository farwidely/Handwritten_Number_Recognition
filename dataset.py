from torch.utils.data import Subset
import torchvision

# 准备数据集
# 训练集
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
# 测试集
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

train_data = Subset(train_dataset, indices=range(0, 6000))
test_data = Subset(test_dataset, indices=range(0, 1000))

for i in range(1000):
    print(train_data[i])

# 查看数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为: {train_data_size}")
print(f"测试数据集的长度为: {test_data_size}")
