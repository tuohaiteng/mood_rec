import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from mood_net import EmotionClassifier
class_names=['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
# 定义推理函数
from PIL import Image
import torch
from torchvision import transforms


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


# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier()
model.load_state_dict(torch.load('./weights/face_20240804_023840.pth'))
model.to(device)

# 推理单张图片
image_path = './data/mooddata/test/00003.png'  # 替换为你要推理的图片路径
label = infer_single_image(model, image_path, class_names, device)
print(f'{os.path.basename(image_path)}  {label}')