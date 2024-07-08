"""
训练模型：
1. 分标签
2. 把数据集里的图片(及标签)按比例分成训练集和验证集，
   图片和标签一定要完全对应，即对应图片和标签都应处于训练集或者数据集
3. 训练模型
"""
import time
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch

print("是否使用GPU训练：{}".format(torch.cuda.is_available()))
if torch.cuda.is_available:
    print("GPU名称为：{}".format(torch.cuda.get_device_name()))

# 数据增强太多也可能造成训练出不好的结果，而且耗时长，宜增强两三倍即可
normalize = transforms.Normalize(mean=[.5,.5,.5], std=[.5,.5,.5]) # 规范化
transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(), normalize]) # 数据处理

dataset_train = ImageFolder('D:/Desktop/photo_data/train', transform=transform) # 训练数据集
dataset_valid = ImageFolder('D:/Desktop/photo_data/valid', transform=transform) # 测试数据集
# 看一下类别和索引
print(dataset_train.class_to_idx)
print(dataset_valid.class_to_idx)

train_data_size = len(dataset_train)
test_data_size = len(dataset_valid)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# torch自带的标准数据集加载函数
dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
dataloader_test = DataLoader(dataset_valid, batch_size=4, shuffle=True, num_workers=0, drop_last=True)

# 模型加载
model_ft = models.resnet18(pretrained=True) #使用迁移学习，加载预训练权重

in_features = model_ft.fc.in_features
model_ft.fc = nn. Sequential(nn.Linear(in_features, 36),
                             nn.Linear(36,3)) # 将最后的全连接改为（36，3），使输出为六个小数，对应3个类别的置信度

model_ft = model_ft.cuda() # 将模型迁移到gpu

# 优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda() # 将loss迁移到gpu
learn_rate = 0.01 # 设置学习率
optimizer = torch.optim.SGD(model_ft.parameters(), lr=learn_rate, momentum=0.01) # 可调超参数

total_train_step = 0
total_test_step = 0
epoch = 50 # 迭代次数
writer = SummaryWriter("logs_train_wyz")
best_acc = -1
ss_time = time.time()

for i in range(epoch):
    start_time = time.time()
    print("--------第{}轮训练开始-------------".format(i+1))
    model_ft.train()
    for data in dataloader_train:
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        outputs = model_ft(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad() # 梯度归零
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 梯度优化

        total_train_step = total_train_step+1
        if total_train_step%100 == 0:
            end_time = time.time()
            print("使用GPU训练100次的时间为：{}".format(end_time-start_time))
            print("训练次数：{}, loss：{}".format(total_train_step, loss.item()))
    model_ft.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in dataloader_test:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model_ft(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy = total_accuracy+accuracy
        print("整体测试集上的loss：{}(越小越好，与上面的loss无关此为测试集的总loss)".format(total_test_loss))
        print("整体测试集上的正确率：{}(越大越好)".format(total_accuracy/len(dataset_valid)))

        writer.add_scalar("valid_loss", (total_accuracy/len(dataset_valid)),(i+1)) # 选择性使用哪一个
        total_test_step = total_test_step + 1
        if total_accuracy > best_acc:  # 保存迭代次数中最好的模型
            print("已修改模型")
            best_acc = total_accuracy
            torch.save(model_ft, "best_model_wyz.pth")
ee_time = time.time()
zong_time = ee_time - ss_time
print("训练总共用时：{}h:{}m:{}s".format(int(zong_time//3600), int((zong_time%3600)//60), int(zong_time%60)))
writer.close()








