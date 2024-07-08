import os
import torch
import torchvision
import cv2
from PIL import Image
from torch import nn


i = 0 # 识别图片计数
root_path = "D:/Desktop/photo_org/baby"
names = os.listdir(root_path)
for name in names:
    print(name)
    i = i + 1
    data_class = ["baby", "cat", "dafeifei"]
    image_path = os.path.join(root_path, name)
    image_org = Image.open(image_path)
    print(image_org)
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)),
                                                 torchvision.transforms.ToTensor()])
    image = transforms(image_org)
    print(image.shape)

    model_ft = torchvision.models.resnet18() # 需要使用训练时的相同模型
    in_features = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(in_features, 36),
                                nn.Linear(36,3))
    
    # 选择训练后得到的模型文件
    model = torch.load("best_model_wyz.pth", map_location = torch.device("cpu"))
    image = torch.reshape(image, (1,3,64,64)) # 修改待预测图片尺寸，需要与训练时一致
    model.eval()

    with torch.no_grad():
        output = model(image)
    label_class = data_class[int(output.argmax(1))]
    print(output)
    print("第{}张图片预测为：{}".format(i, label_class))

    # 把图片按类别放入文件夹
    output_dir = "D:/Desktop/photo_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    class_dir = os.path.join(output_dir, str(label_class))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    img_out_path = os.path.join(class_dir, f"image_{i}.jpg")
    
    image_output = cv2.imread(image_path)
    if image_output is None:
        print(f"无法读取图片{name}.jpg")
    cv2.imwrite(img_out_path, image_output)
    
    

