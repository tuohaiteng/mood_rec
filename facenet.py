import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from datetime import datetime
from mood_net import EmotionClassifier
from torch.optim.lr_scheduler import ReduceLROnPlateau

def infer_single_image(model, img, class_names, device):
    """
    对单张图像进行推理。

    参数:
    - model: 预训练的情感分类模型
    - img: PIL.Image 对象，表示要推理的图像
    - class_names: 情感类别名称列表
    - device: PyTorch 设备（'cpu' 或 'cuda'）

    返回:
    - label: 预测的情感类别
    """
    # 确保图像是灰度格式
    img = img.convert('L')

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 灰度图像的均值和标准差
    ])

    # 应用预处理
    img = transform(img)
    img = img.unsqueeze(0)  # 增加批量维度
    img = img.to(device)

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]

    return label

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

# 定义数据集路径
TRAINING_DIR = './data/mooddata/train/'
TEST_DIR = './data/mooddata/test/'

# 数据预处理
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 加载数据集
dataset = datasets.ImageFolder(root=TRAINING_DIR, transform=train_transform)

# 数据集划分
validation_split = 0.25
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
np.random.shuffle(indices)
train_indices, valid_indices = indices[split:], indices[:split]

# 打印数据集划分
print(f'数据集总样本数: {dataset_size}')
print(f'训练样本数: {len(train_indices)}')
print(f'验证样本数: {len(valid_indices)}')

# 创建数据加载器
train_loader = DataLoader(dataset, batch_size=64, sampler=SubsetRandomSampler(train_indices))
valid_loader = DataLoader(dataset, batch_size=64, sampler=SubsetRandomSampler(valid_indices))

# 打印训练和验证数据集的一些样本
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_batch(data_loader, title):
    images, labels = next(iter(data_loader))
    grid_img = make_grid(images)
    plt.figure(figsize=(12, 6))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()

print("训练数据集的一些样本：")
show_batch(train_loader, 'Train Data Samples')
print("验证数据集的一些样本：")
show_batch(valid_loader, 'Validation Data Samples')


# 打印类别名称
class_names = dataset.classes
print("类别名称:", class_names)

# 模型定义和训练设置
model = EmotionClassifier()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("使用设备:", device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

num_epochs = 150
train_loss_history = []
valid_loss_history = []
train_acc_history = []
valid_acc_history = []

# 训练循环
print("开始训练!")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct_preds += torch.sum(preds == labels.data)
        total_preds += labels.size(0)

    epoch_train_loss = running_loss / len(train_loader.sampler)
    epoch_train_acc = correct_preds.double() / total_preds
    train_loss_history.append(epoch_train_loss)
    train_acc_history.append(epoch_train_acc)

    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_preds += labels.size(0)

    epoch_valid_loss = running_loss / len(valid_loader.sampler)
    epoch_valid_acc = correct_preds.double() / total_preds
    valid_loss_history.append(epoch_valid_loss)
    valid_acc_history.append(epoch_valid_acc)

    print(f'第{epoch + 1}/{num_epochs}轮, '
          f'训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_acc:.4f}, '
          f'验证损失: {epoch_valid_loss:.4f}, 验证准确率: {epoch_valid_acc:.4f}')

    # 更新学习率
    scheduler.step(epoch_valid_loss)


# 保存模型
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = './weights'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, f'face_{timestamp}.pth')
torch.save(model.state_dict(), save_path)
print(f'模型已保存至 {save_path}')

# 推理函数


# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier()
model.load_state_dict(torch.load('./weights/face.pth'))
model.to(device)

# 推理单张图片
image_path = './data/mooddata/test/00001.png'
label = infer_single_image(model, image_path, class_names, device)
print(f'{os.path.basename(image_path)}  {label}')
