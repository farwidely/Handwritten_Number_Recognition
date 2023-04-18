import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 设置计算硬件为cpu或cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集, 对数据集进行归一化
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
# 训练集
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# 测试集
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# 查看数据集长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print(f"训练数据集的长度为: {train_data_size}")
print(f"测试数据集的长度为: {test_data_size}")

# 设置Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)


class MyMNIST(nn.Module):
    def __init__(self):
        super(MyMNIST, self).__init__()
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


# 初始化模型
model = MyMNIST()
model.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2
momentum = 5e-1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 设置训练网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 20

# 添加tensorboard
writer = SummaryWriter("log_train")

start = time.time()

for i in range(epoch):
    print(f"------第 {i + 1} 轮训练开始------")

    start1 = time.time()

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

    end1 = time.time()
    print(f"本轮训练时长为{end1 - start1}秒")
    start2 = time.time()

    # 测试步骤开始
    model.eval()

    # 初始化模型在训练集上的评价指标变量
    total_train_loss = 0
    total_train_accuracy = 0
    total_train_tp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    total_train_fp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    total_train_fn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # 初始化模型在测试集上的评价指标变量
    total_test_loss = 0
    total_test_accuracy = 0
    total_test_tp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    total_test_fp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    total_test_fn = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    with torch.no_grad():
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            train_loss = loss_fn(outputs, targets)
            total_train_loss += train_loss.item()
            train_accuracy = (outputs.argmax(1) == targets).sum()
            total_train_accuracy += train_accuracy

            # 计算训练集混淆矩阵
            CM_train = confusion_matrix(outputs.argmax(1).to("cpu"), targets.to("cpu"), labels=[0, 1, 2, 3, 4, 5, 6, 7,
                                                                                                8, 9])

            TP = np.diag(CM_train)
            FP = CM_train.sum(axis=0) - np.diag(CM_train)
            FN = CM_train.sum(axis=1) - np.diag(CM_train)
            total_train_tp += TP
            total_train_fp += FP
            total_train_fn += FN

        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            test_loss = loss_fn(outputs, targets)
            total_test_loss += test_loss.item()
            test_accuracy = (outputs.argmax(1) == targets).sum()
            total_test_accuracy += test_accuracy

            # 计算测试集混淆矩阵
            CM_test = confusion_matrix(outputs.argmax(1).to("cpu"), targets.to("cpu"), labels=[0, 1, 2, 3, 4, 5, 6, 7,
                                                                                               8, 9])

            TP = np.diag(CM_test)
            FP = CM_test.sum(axis=0) - np.diag(CM_test)
            FN = CM_test.sum(axis=1) - np.diag(CM_test)
            total_test_tp += TP
            total_test_fp += FP
            total_test_fn += FN

    # 计算训练集查准率、查全率、F1指数
    train_Precision = total_train_tp / (total_train_tp + total_train_fp)
    train_Recall = total_train_tp / (total_train_tp + total_train_fn)
    train_f1 = 2 * train_Precision * train_Recall / (train_Precision + train_Recall)

    print(f"整体训练集上的Loss: {total_train_loss}")
    print(f"整体训练集上的正确率: {total_train_accuracy / train_data_size}")
    print(f"label-0在训练集的查准率: {train_Precision[0]}")
    print(f"label-0在训练集的查全率: {train_Recall[0]}")
    print(f"label-0在训练集的F1-score: {train_f1[0]}")
    print(f"label-1在训练集的查准率: {train_Precision[1]}")
    print(f"label-1在训练集的查全率: {train_Recall[1]}")
    print(f"label-1在训练集的F1-score: {train_f1[1]}")
    print(f"label-2在训练集的查准率: {train_Precision[2]}")
    print(f"label-2在训练集的查全率: {train_Recall[2]}")
    print(f"label-2在训练集的F1-score: {train_f1[2]}")
    print(f"label-3在训练集的查准率: {train_Precision[3]}")
    print(f"label-3在训练集的查全率: {train_Recall[3]}")
    print(f"label-3在训练集的F1-score: {train_f1[3]}")
    print(f"label-4在训练集的查准率: {train_Precision[4]}")
    print(f"label-4在训练集的查全率: {train_Recall[4]}")
    print(f"label-4在训练集的F1-score: {train_f1[4]}")
    print(f"label-5在训练集的查准率: {train_Precision[5]}")
    print(f"label-5在训练集的查全率: {train_Recall[5]}")
    print(f"label-5在训练集的F1-score: {train_f1[5]}")
    print(f"label-6在训练集的查准率: {train_Precision[6]}")
    print(f"label-6在训练集的查全率: {train_Recall[6]}")
    print(f"label-6在训练集的F1-score: {train_f1[6]}")
    print(f"label-7在训练集的查准率: {train_Precision[7]}")
    print(f"label-7在训练集的查全率: {train_Recall[7]}")
    print(f"label-7在训练集的F1-score: {train_f1[7]}")
    print(f"label-8在训练集的查准率: {train_Precision[8]}")
    print(f"label-8在训练集的查全率: {train_Recall[8]}")
    print(f"label-8在训练集的F1-score: {train_f1[8]}")
    print(f"label-9在训练集的查准率: {train_Precision[9]}")
    print(f"label-9在训练集的查全率: {train_Recall[9]}")
    print(f"label-9在训练集的F1-score: {train_f1[9]}")

    # 将训练集结果载入tensorboard
    writer.add_scalar("train_loss", total_train_loss, total_test_step)
    writer.add_scalar("train_accuracy", total_train_accuracy / train_data_size, total_test_step)
    writer.add_scalar("label-0_train_precision", train_Precision[0], total_test_step)
    writer.add_scalar("label-0_train_recall", train_Recall[0], total_test_step)
    writer.add_scalar("label-0_train_F1-score", train_f1[0], total_test_step)
    writer.add_scalar("label-1_train_precision", train_Precision[1], total_test_step)
    writer.add_scalar("label-1_train_recall", train_Recall[1], total_test_step)
    writer.add_scalar("label-1_train_F1-score", train_f1[1], total_test_step)
    writer.add_scalar("label-2_train_precision", train_Precision[2], total_test_step)
    writer.add_scalar("label-2_train_recall", train_Recall[2], total_test_step)
    writer.add_scalar("label-2_train_F1-score", train_f1[2], total_test_step)
    writer.add_scalar("label-3_train_precision", train_Precision[3], total_test_step)
    writer.add_scalar("label-3_train_recall", train_Recall[3], total_test_step)
    writer.add_scalar("label-3_train_F1-score", train_f1[3], total_test_step)
    writer.add_scalar("label-4_train_precision", train_Precision[4], total_test_step)
    writer.add_scalar("label-4_train_recall", train_Recall[4], total_test_step)
    writer.add_scalar("label-4_train_F1-score", train_f1[4], total_test_step)
    writer.add_scalar("label-5_train_precision", train_Precision[5], total_test_step)
    writer.add_scalar("label-5_train_recall", train_Recall[5], total_test_step)
    writer.add_scalar("label-5_train_F1-score", train_f1[5], total_test_step)
    writer.add_scalar("label-6_train_precision", train_Precision[6], total_test_step)
    writer.add_scalar("label-6_train_recall", train_Recall[6], total_test_step)
    writer.add_scalar("label-6_train_F1-score", train_f1[6], total_test_step)
    writer.add_scalar("label-7_train_precision", train_Precision[7], total_test_step)
    writer.add_scalar("label-7_train_recall", train_Recall[7], total_test_step)
    writer.add_scalar("label-7_train_F1-score", train_f1[7], total_test_step)
    writer.add_scalar("label-8_train_precision", train_Precision[8], total_test_step)
    writer.add_scalar("label-8_train_recall", train_Recall[8], total_test_step)
    writer.add_scalar("label-8_train_F1-score", train_f1[8], total_test_step)
    writer.add_scalar("label-9_train_precision", train_Precision[9], total_test_step)
    writer.add_scalar("label-9_train_recall", train_Recall[9], total_test_step)
    writer.add_scalar("label-9_train_F1-score", train_f1[9], total_test_step)

    # 计算测试集查准率、查全率、F1指数
    test_Precision = total_test_tp / (total_test_tp + total_test_fp)
    test_Recall = total_test_tp / (total_test_tp + total_test_fn)
    test_f1 = 2 * test_Precision * test_Recall / (test_Precision + test_Recall)

    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_test_accuracy / test_data_size}")
    print(f"label-0在测试集的查准率: {test_Precision[0]}")
    print(f"label-0在测试集的查全率: {test_Recall[0]}")
    print(f"label-0在测试集的F1-score: {test_f1[0]}")
    print(f"label-1在测试集的查准率: {test_Precision[1]}")
    print(f"label-1在测试集的查全率: {test_Recall[1]}")
    print(f"label-1在测试集的F1-score: {test_f1[1]}")
    print(f"label-2在测试集的查准率: {test_Precision[2]}")
    print(f"label-2在测试集的查全率: {test_Recall[2]}")
    print(f"label-2在测试集的F1-score: {test_f1[2]}")
    print(f"label-3在测试集的查准率: {test_Precision[3]}")
    print(f"label-3在测试集的查全率: {test_Recall[3]}")
    print(f"label-3在测试集的F1-score: {test_f1[3]}")
    print(f"label-4在测试集的查准率: {test_Precision[4]}")
    print(f"label-4在测试集的查全率: {test_Recall[4]}")
    print(f"label-4在测试集的F1-score: {test_f1[4]}")
    print(f"label-5在测试集的查准率: {test_Precision[5]}")
    print(f"label-5在测试集的查全率: {test_Recall[5]}")
    print(f"label-5在测试集的F1-score: {test_f1[5]}")
    print(f"label-6在测试集的查准率: {test_Precision[6]}")
    print(f"label-6在测试集的查全率: {test_Recall[6]}")
    print(f"label-6在测试集的F1-score: {test_f1[6]}")
    print(f"label-7在测试集的查准率: {test_Precision[7]}")
    print(f"label-7在测试集的查全率: {test_Recall[7]}")
    print(f"label-7在测试集的F1-score: {test_f1[7]}")
    print(f"label-8在测试集的查准率: {test_Precision[8]}")
    print(f"label-8在测试集的查全率: {test_Recall[8]}")
    print(f"label-8在测试集的F1-score: {test_f1[8]}")
    print(f"label-9在测试集的查准率: {test_Precision[9]}")
    print(f"label-9在测试集的查全率: {test_Recall[9]}")
    print(f"label-9在测试集的F1-score: {test_f1[9]}")

    # 将测试集结果载入tensorboard
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_test_accuracy / test_data_size, total_test_step)
    writer.add_scalar("label-0_test_precision", test_Precision[0], total_test_step)
    writer.add_scalar("label-0_test_recall", test_Recall[0], total_test_step)
    writer.add_scalar("label-0_test_F1-score", test_f1[0], total_test_step)
    writer.add_scalar("label-1_test_precision", test_Precision[1], total_test_step)
    writer.add_scalar("label-1_test_recall", test_Recall[1], total_test_step)
    writer.add_scalar("label-1_test_F1-score", test_f1[1], total_test_step)
    writer.add_scalar("label-2_test_precision", test_Precision[2], total_test_step)
    writer.add_scalar("label-2_test_recall", test_Recall[2], total_test_step)
    writer.add_scalar("label-2_test_F1-score", test_f1[2], total_test_step)
    writer.add_scalar("label-3_test_precision", test_Precision[3], total_test_step)
    writer.add_scalar("label-3_test_recall", test_Recall[3], total_test_step)
    writer.add_scalar("label-3_test_F1-score", test_f1[3], total_test_step)
    writer.add_scalar("label-4_test_precision", test_Precision[4], total_test_step)
    writer.add_scalar("label-4_test_recall", test_Recall[4], total_test_step)
    writer.add_scalar("label-4_test_F1-score", test_f1[4], total_test_step)
    writer.add_scalar("label-5_test_precision", test_Precision[5], total_test_step)
    writer.add_scalar("label-5_test_recall", test_Recall[5], total_test_step)
    writer.add_scalar("label-5_test_F1-score", test_f1[5], total_test_step)
    writer.add_scalar("label-6_test_precision", test_Precision[6], total_test_step)
    writer.add_scalar("label-6_test_recall", test_Recall[6], total_test_step)
    writer.add_scalar("label-6_test_F1-score", test_f1[6], total_test_step)
    writer.add_scalar("label-7_test_precision", test_Precision[7], total_test_step)
    writer.add_scalar("label-7_test_recall", test_Recall[7], total_test_step)
    writer.add_scalar("label-7_test_F1-score", test_f1[7], total_test_step)
    writer.add_scalar("label-8_test_precision", test_Precision[8], total_test_step)
    writer.add_scalar("label-8_test_recall", test_Recall[8], total_test_step)
    writer.add_scalar("label-8_test_F1-score", test_f1[8], total_test_step)
    writer.add_scalar("label-9_test_precision", test_Precision[9], total_test_step)
    writer.add_scalar("label-9_test_recall", test_Recall[9], total_test_step)
    writer.add_scalar("label-9_test_F1-score", test_f1[9], total_test_step)

    end2 = time.time()
    print(f"本轮测试时长为{end2 - start2}秒\n")

    total_test_step += 1

    if i == 19:
        torch.save(model, f"./trained_models/model_gpu_{i + 1}.pth")
        print("模型已保存")

end = time.time()
print(f"训练+测试总时长为{end - start}秒")
print("学号：221115194    姓名：邓广远")

writer.close()
