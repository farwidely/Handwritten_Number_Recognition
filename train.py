from torch.utils.data import Subset, DataLoader
# import torchvision
from models import *

# 设置计算硬件为cpu或cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
# 训练集
train_data = torchvision.datasets.MNIST(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 测试集
test_data = torchvision.datasets.MNIST(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# train_data = Subset(train_dataset, indices=range(0, 6000))
# test_data = Subset(test_dataset, indices=range(0, 1000))

# 查看数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为: {train_data_size}")
print(f"测试数据集的长度为: {test_data_size}")

# 设置Dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 初始化模型
model = MyMNIST()
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 50

for i in range(epoch):
    print(f"------第 {i+1} 轮训练开始------")

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print(f"训练次数: {total_train_step}，Loss: {loss.item()}")

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy/test_data_size}")

    total_test_step += 1

    if i == 49:
        torch.save(model, f"./trained_models/model_gpu_{i+1}.pth")
        print("模型已保存")