项目可分为两个部分：人脸检测和情绪识别

# 人脸检测

## 模型介绍

人脸检测部分使用YOLOv5-Face模型，

YOLOv5-Face在YOLOv5的基础上添加了一个 5-Point Landmark Regression Head（关键点回归），并对Landmark Regression Head使用了Wing loss进行约束。


## 模型结构图

YOLOv5Face是以YOLOv5作为Baseline来进行改进和再设计以适应人脸检测

### YOLOv5结构图

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/73c2da4db01a49bc9b2a8b046f4c795e.png)

### YOLOv5-face结构图

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b2b169ff625c4301b1bc7440c2c94edf.png)
## 数据集

Wider Face数据集最早是在2015年公开的。该数据集的图片来源是WIDER数据集，从中挑选出了32,203图片并进行了人脸标注，总共标注了393,703个人脸数据。并且对于每张人脸都附带有更加详细的信息，包扩blur（模糊程度）, expression（表情）, illumination（光照）, occlusion（遮挡）, pose（姿态）等
在数据集中，根据事件场景的类型分为了61个类。接着根据每个类别按照40% / 10% / 50%的比例划分到训练集，验证集以及测试集中。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d20a4a4b20d04dfb86c1f6784d620b9f.png)

# 情绪识别

## 模型介绍

情绪识别部分采用一个用于情感分类的卷积神经网络（CNN）。它的架构如下：

## EmotionClassifier 模型架构

1. **输入层**：接受大小为 `1x48x48` 的灰度图像。
2. 卷积层1：
   - **操作**：应用 32 个大小为 `3x3` 的卷积核
   - **输出**：`32x48x48`
3. **ReLU激活函数**：非线性激活。
4. 最大池化层1：
   - **操作**：池化窗口大小为 `2x2`
   - **输出**：`32x24x24`
5. 卷积层2：
   - **操作**：应用 64 个大小为 `3x3` 的卷积核
   - **输出**：`64x24x24`
6. **ReLU激活函数**：非线性激活。
7. 最大池化层2：
   - **操作**：池化窗口大小为 `2x2`
   - **输出**：`64x12x12`
8. 卷积层3：
   - **操作**：应用 128 个大小为 `3x3` 的卷积核
   - **输出**：`128x12x12`
9. **ReLU激活函数**：非线性激活。
10. 最大池化层3：
    - **操作**：池化窗口大小为 `2x2`
    - **输出**：`128x6x6`
11. 展平层：
    - **操作**：将特征图展平为一维向量
    - **输出**：`128*6*6`
12. 全连接层1：
    - **神经元数量**：256
13. **ReLU激活函数**：非线性激活。
14. Dropout层：
    - **丢弃概率**：50%
15. 全连接层2：
    - **神经元数量**：7
16. 输出层：
    - **输出**：表示情感分类结果。

这种架构逐步提取图像特征，最后通过全连接层和输出层进行情感分类
## 模型结构图

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 48, 48]             320
         MaxPool2d-2           [-1, 32, 24, 24]               0
            Conv2d-3           [-1, 64, 24, 24]          18,496
         MaxPool2d-4           [-1, 64, 12, 12]               0
            Conv2d-5          [-1, 128, 12, 12]          73,856
         MaxPool2d-6            [-1, 128, 6, 6]               0
            Linear-7                  [-1, 256]       1,179,904
           Dropout-8                  [-1, 256]               0
            Linear-9                    [-1, 7]           1,799
================================================================
Total params: 1,274,375
Trainable params: 1,274,375
Non-trainable params: 0
----------------------------------------------------------------

```



## 数据集

**FER**数据集，其中图像已被重新标记为 7 种情绪类型之一：中性、快乐、惊讶、悲伤、愤怒、厌恶、恐惧，尺寸限制为 48×48

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8bd8903b4be475e8f63211a84c5b2d9.png)

# 主要过程

detect_mood.py主要过程如下：

1. **导入库和定义函数**：
   - 导入必要的库和模块，包括PyTorch、OpenCV、以及自定义的模型和工具函数。
   - 定义了`plot_one_box`函数用于在图像上绘制检测框。
2. **情绪分类模型**：
   - 定义了一个名为`EmotionClassifier`的神经网络模型，用于对面部图像进行情绪分类。该模型由多个卷积层、池化层和全连接层组成。
3. **图像推理函数**：
   - `infer_single_image`函数加载并处理单张图像，进行模型推理并返回预测的情绪标签。
4. **检测和分类主函数**：
   - `detect`函数设置了检测和分类的参数，包括模型权重、图像大小、阈值等。
   - 加载YOLOv5面部检测模型和情绪分类模型。
   - 根据输入来源（视频流或图像文件）初始化数据加载器。
   - 对每张图像或每帧视频进行处理，进行面部检测和情绪分类，绘制检测框和情绪标签。
   - 保存结果到指定的文件夹中，并在需要时显示结果图像。
5. **主程序**：
   - 使用`argparse`解析命令行参数，调用`detect`函数进行实际的检测和分类任务。

整体流程包括图像的加载与处理、面部检测、情绪分类、结果绘制与保存。

## 关键函数

```python
def apply_classifier(x, model, img, im0):
    class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    将第二阶段分类器应用于YOLO的输出。

    参数：
    - x: YOLO检测结果的列表
    - model: 分类器模型
    - img: 以YOLO期望格式输入的图像
    - im0: 原始图像或原始图像列表

    返回：
    - x: 修改后的YOLO检测结果列表
    - face_emotions: 检测到的人脸情绪
    """
    im0 = [im0] if isinstance(im0, np.ndarray) else im0  # 确保 im0 是一个列表
    face_emotions = []

    for i, d in enumerate(x):  # 遍历每张图片的检测结果
        if d is not None and len(d):
            d = d.clone()

            # 重新调整和填充切割图像
            b = xyxy2xywh(d[:, :4])  # 将框转换为中心点-宽高格式
            d[:, :4] = xywh2xyxy(b).long()  # 再转换回左上角-右下角格式
            # 按原始图像大小缩放坐标
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
            ims = []

            for j, a in enumerate(d):  # 遍历每个检测结果
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]  # 提取检测区域
                im = Image.fromarray(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))  # 转换为PIL图像
                im = im.resize((48, 48))  # 调整图像尺寸为48x48
                emotion = infer_single_image(model, im, class_names, device)  # 预测情绪
                print("emotion:",emotion)
                face_emotions.append(emotion)

    return x, face_emotions

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
```



# 项目运行

运行detect_mood.py后，项目自动识别demo/dec_images文件夹中图片，将检测结果保存到rec_result文件夹中

## 结果展示

项目可分为两个部分：人脸检测和情绪识别

# 人脸检测

## 模型介绍

人脸检测部分使用YOLOv5-Face模型，

YOLOv5-Face在YOLOv5的基础上添加了一个 5-Point Landmark Regression Head（关键点回归），并对Landmark Regression Head使用了Wing loss进行约束。


## 模型结构图

YOLOv5Face是以YOLOv5作为Baseline来进行改进和再设计以适应人脸检测

### YOLOv5结构图

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/73c2da4db01a49bc9b2a8b046f4c795e.png)

### YOLOv5-face结构图

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b2b169ff625c4301b1bc7440c2c94edf.png)

## 数据集

Wider Face数据集最早是在2015年公开的。该数据集的图片来源是WIDER数据集，从中挑选出了32,203图片并进行了人脸标注，总共标注了393,703个人脸数据。并且对于每张人脸都附带有更加详细的信息，包扩blur（模糊程度）, expression（表情）, illumination（光照）, occlusion（遮挡）, pose（姿态）等
在数据集中，根据事件场景的类型分为了61个类。接着根据每个类别按照40% / 10% / 50%的比例划分到训练集，验证集以及测试集中。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d20a4a4b20d04dfb86c1f6784d620b9f.png)

# 情绪识别

## 模型介绍

情绪识别部分采用一个用于情感分类的卷积神经网络（CNN）。它的架构如下：

## EmotionClassifier 模型架构

1. **输入层**：接受大小为 `1x48x48` 的灰度图像。
2. 卷积层1：
   - **操作**：应用 32 个大小为 `3x3` 的卷积核
   - **输出**：`32x48x48`
3. **ReLU激活函数**：非线性激活。
4. 最大池化层1：
   - **操作**：池化窗口大小为 `2x2`
   - **输出**：`32x24x24`
5. 卷积层2：
   - **操作**：应用 64 个大小为 `3x3` 的卷积核
   - **输出**：`64x24x24`
6. **ReLU激活函数**：非线性激活。
7. 最大池化层2：
   - **操作**：池化窗口大小为 `2x2`
   - **输出**：`64x12x12`
8. 卷积层3：
   - **操作**：应用 128 个大小为 `3x3` 的卷积核
   - **输出**：`128x12x12`
9. **ReLU激活函数**：非线性激活。
10. 最大池化层3：
    - **操作**：池化窗口大小为 `2x2`
    - **输出**：`128x6x6`
11. 展平层：
    - **操作**：将特征图展平为一维向量
    - **输出**：`128*6*6`
12. 全连接层1：
    - **神经元数量**：256
13. **ReLU激活函数**：非线性激活。
14. Dropout层：
    - **丢弃概率**：50%
15. 全连接层2：
    - **神经元数量**：7
16. 输出层：
    - **输出**：表示情感分类结果。

这种架构逐步提取图像特征，最后通过全连接层和输出层进行情感分类

## 模型结构图

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 48, 48]             320
         MaxPool2d-2           [-1, 32, 24, 24]               0
            Conv2d-3           [-1, 64, 24, 24]          18,496
         MaxPool2d-4           [-1, 64, 12, 12]               0
            Conv2d-5          [-1, 128, 12, 12]          73,856
         MaxPool2d-6            [-1, 128, 6, 6]               0
            Linear-7                  [-1, 256]       1,179,904
           Dropout-8                  [-1, 256]               0
            Linear-9                    [-1, 7]           1,799
================================================================
Total params: 1,274,375
Trainable params: 1,274,375
Non-trainable params: 0
----------------------------------------------------------------

```



## 数据集

**FER**数据集，其中图像已被重新标记为 7 种情绪类型之一：中性、快乐、惊讶、悲伤、愤怒、厌恶、恐惧，尺寸限制为 48×48

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d8bd8903b4be475e8f63211a84c5b2d9.png)

# 主要过程

detect_mood.py主要过程如下：

1. **导入库和定义函数**：
   - 导入必要的库和模块，包括PyTorch、OpenCV、以及自定义的模型和工具函数。
   - 定义了`plot_one_box`函数用于在图像上绘制检测框。
2. **情绪分类模型**：
   - 定义了一个名为`EmotionClassifier`的神经网络模型，用于对面部图像进行情绪分类。该模型由多个卷积层、池化层和全连接层组成。
3. **图像推理函数**：
   - `infer_single_image`函数加载并处理单张图像，进行模型推理并返回预测的情绪标签。
4. **检测和分类主函数**：
   - `detect`函数设置了检测和分类的参数，包括模型权重、图像大小、阈值等。
   - 加载YOLOv5面部检测模型和情绪分类模型。
   - 根据输入来源（视频流或图像文件）初始化数据加载器。
   - 对每张图像或每帧视频进行处理，进行面部检测和情绪分类，绘制检测框和情绪标签。
   - 保存结果到指定的文件夹中，并在需要时显示结果图像。
5. **主程序**：
   - 使用`argparse`解析命令行参数，调用`detect`函数进行实际的检测和分类任务。

整体流程包括图像的加载与处理、面部检测、情绪分类、结果绘制与保存。

## 关键函数

```python
def apply_classifier(x, model, img, im0):
    class_names = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    将第二阶段分类器应用于YOLO的输出。

    参数：
    - x: YOLO检测结果的列表
    - model: 分类器模型
    - img: 以YOLO期望格式输入的图像
    - im0: 原始图像或原始图像列表

    返回：
    - x: 修改后的YOLO检测结果列表
    - face_emotions: 检测到的人脸情绪
    """
    im0 = [im0] if isinstance(im0, np.ndarray) else im0  # 确保 im0 是一个列表
    face_emotions = []

    for i, d in enumerate(x):  # 遍历每张图片的检测结果
        if d is not None and len(d):
            d = d.clone()

            # 重新调整和填充切割图像
            b = xyxy2xywh(d[:, :4])  # 将框转换为中心点-宽高格式
            d[:, :4] = xywh2xyxy(b).long()  # 再转换回左上角-右下角格式
            # 按原始图像大小缩放坐标
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)
            ims = []

            for j, a in enumerate(d):  # 遍历每个检测结果
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]  # 提取检测区域
                im = Image.fromarray(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))  # 转换为PIL图像
                im = im.resize((48, 48))  # 调整图像尺寸为48x48
                emotion = infer_single_image(model, im, class_names, device)  # 预测情绪
                print("emotion:",emotion)
                face_emotions.append(emotion)

    return x, face_emotions

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
```



# 项目运行

运行detect_mood.py后，项目自动识别demo/dec_images文件夹中图片，将检测结果保存到rec_result文件夹中

## 结果展示

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9507004c002541338b820be08b7479f9.png)

