import argparse
import torch.backends.cudnn as cudnn
from models.experimental import *
from utils.datasets import *
from utils.utils import *

import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from datetime import datetime
from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    #print("label_zhi",label)
    if label:

        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        image=change_cv2_draw(img,label,(int(c1[0]), int(c1[1]) - 30),5,[225,225,225])
        #print("image:",image)
    return image

def infer_single_image(model, image_path, class_names, device):
    # 加载并处理图像
    img = Image.open(image_path).convert('L')  # 转换为灰度图像
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 适用于灰度图像的均值和标准差
    ])
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


class EmotionClassifier(nn.Module):
    def __init__(self):
        super(EmotionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # 增加卷积核数量，减小核大小
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)  # 增加全连接层的神经元数量
        self.fc2 = nn.Linear(256, 7)  # 输出层保持不变
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def detect(save_img=False):
    # 定义参数
    args = {
        "classify": True,
        "det_weights": "./weights/yolov5-blazeface.pt",
        "rec_weights": "./weights/face_20240804_164437.pth",
        "source": "./demo/dec_images/",
        "output": "demo/rec_result",
        "img_size": 640,
        "conf_thres": 0.5,
        "iou_thres": 0.5,
        "device": "",
        "view_img": False,
        "save_txt": False,
        "classes": None,
        "agnostic_nms": False,
        "augment": False,
        "update": False,
    }
    img_size = 640
    # 创建一个带有参数的 argparse.Namespace 对象
    opt = argparse.Namespace(**args)

    classify, out, source, det_weights, rec_weights, view_img, save_txt, imgsz = \
        opt.classify, opt.output, opt.source, opt.det_weights, opt.rec_weights,  opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete rec_result folder
    os.makedirs(out)  # make new rec_result folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load YOLOv5-face model
    model = attempt_load(det_weights, map_location=device)  # load FP32 model
    print("Load detection pretrained model successful!")
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    print(imgsz)
    # Load emotion recognition model
    if classify:
        modelc = EmotionClassifier().to(device)
        modelc.load_state_dict(torch.load(rec_weights, map_location=torch.device('cpu')))
        print("Load recognition pretrained model successful!")
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size demo
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    print("names:")
    print(names)
    print(colors)
    # Run demo
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    print(imgsz)

    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, im, im0s, vid_cap in dataset:
        # img = torch.from_numpy(img).to(device)
        # img = img.half() if half else img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # if img.ndimension() == 3:
        #     img = img.unsqueeze(0)
        if len(im.shape) == 4:
            orgimg = np.squeeze(im.transpose(0, 2, 3, 1), axis=0)
        else:
            orgimg = im.transpose(1, 2, 0)

        orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)
        img0 = copy.deepcopy(orgimg)
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert from w,h,c to c,w,h
        img = img.transpose(2, 0, 1).copy()

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        print("Inference")
        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        #print(pred)
        # Apply NMS-face
        pred = non_max_suppression_face(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')
        #print(pred)
        # Apply Classifier
        if classify:
            pred, mood_class = apply_classifier(pred, modelc, img, im0s)
        print("Apply Classifier!")
        #print(pred)
        print("mood class:")
        print(mood_class)
        t2 = torch_utils.time_synchronized()


        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            #print("chushi_im0",im0)
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                print("det:")
                print(det)

                # Print results
                unique_classes = det[:, 15].unique()
                print(f"Unique class indices in detections: {unique_classes}")
                for c in unique_classes:
                    n = (det[:, 15] == c).sum()  # detections per class
                    class_index = int(c)  # 确保类索引是整数


                    if 0 <= class_index < len(mood_class):
                        s += '%g %ss, ' % (n, names[0])  # add to string
                        print(f"Detected class index: {class_index}, Total: {n}")  # 调试输出
                    else:
                        print(f"Warning: Class index {class_index} is out of range for names list")

                # Write results
                for de, mood in zip(det, mood_class):
                    # *xyxy, conf, *landmark,cls = de
                    xyxy = de[:4]  # 前 4 个元素，边界框坐标
                    conf = de[4]  # 第 5 个元素，置信度
                    landmarks = de[5:-1]  # 从第 6 个元素到倒数第 2 个元素，地标
                    cls = de[-1]  # 最后一个元素，类别索引

                    print("de:",de)
                    print("conf:", conf)
                    class_index = int(cls)  # 确保类索引是整数

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (class_index, xywh))  # label format

                    if save_img or view_img:  # Add bbox to image

                        lb = mood
                        print("label:")
                        # print(label)
                        label = '%s %.2f' % (lb, conf)
                        print(label)
                       #print("im0:",im0)
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[0], line_thickness=3)
                        #print("plot_im0:",im0)
            # Print time (demo + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            print("save_img:",save_img)
            if save_img:
                if dataset.mode != 'video':
                    #print("im0", im0)
                    cv2.imwrite(save_path, im0)

                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        vid_cap = cv2.VideoCapture(vid_path)
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # rec_result video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


    if opt.update:  # Update all models (to fix SourceChangeWarning)
        for opt.det_weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
            detect()

    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify', nargs='+', type=str, default=True, help='True rec')
    parser.add_argument('--det-weights', nargs='+', type=str, default='./weights/yolov5-blazeface.pt',
                        help='model.pt path(s)')
    parser.add_argument('--rec-weights', nargs='+', type=str, default='./weights/face_20240804_164437.pth',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./demo/dec_images/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='demo/rec_result', help='rec_result folder')  # rec_result folder
    parser.add_argument('--img-size', type=int, default=640, help='demo size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented demo')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print("opt:")
    print(opt)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
                detect(save_img=True)
                create_pretrained(opt.weights, opt.weights)
        else:
            detect(save_img=True)

