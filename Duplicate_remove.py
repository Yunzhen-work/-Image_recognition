"""
删除文件夹中的重复图片
"""

import shutil
import numpy as np
from PIL import Image
import os

# 定义辅助函数判断两张图片是否相同
def 比较图片大小 (dir_image1, dir_image2):
    with open (dir_image1, "rb") as f1:
        size1 = len(f1.read())
    with open (dir_image2, "rb") as f2:
        size2 = len(f2.read())
    if size1 == size2:
        result = "图片大小相同"
    else:
        result = "图片大小不同"
    return result

def 比较图片尺寸 (dir_image1, dir_image2):
    image1 = Image.open(dir_image1)
    image2 = Image.open(dir_image2)
    if image1.size == image2.size:
        result = "图片尺寸相同"
    else:
        result = "图片尺寸不同"
    return result

def 比较图片内容(dir_image1, dir_image2):
    image1 = np.array(Image.open(dir_image1))
    image2 = np.array(Image.open(dir_image2))
    if(np.array_equal(image1, image2)):
        result = "图片内容相同"
    else:
        result = "图片内容不同"
    return result

def 比较两张图片是否相同(dir_image1, dir_image2):
    # 依次比较以下几个方面：大小、长宽、图片内容，如果前一个不同，图片不同
    result = "两张图片不相同"
    大小 = 比较图片大小(dir_image1, dir_image2)   
    if 大小 == "图片大小相同":
        尺寸 = 比较图片尺寸(dir_image1, dir_image2)
        if 尺寸 == "图片尺寸相同":
            内容 = 比较图片内容(dir_image1, dir_image2)
            if 内容 == "图片内容相同":
                result = "两张图片相同"
    return result


# 判断文件夹内是否有重复图片
if __name__ == '__main__':

    load_path = 'C:/Users/weiyu/Desktop/photo_org' # 要去重的图片
    save_path = 'C:/Users/weiyu/Desktop/photo_dup' # 空文件夹，用于存储检测到的重复图片
    # exist_ok = True 意思是如果文件夹已经存在，不要报错
    os.makedirs(save_path, exist_ok=True)

    # 获取图片列表 file_map
    file_map = {}
    image_size = 0
    # 用os.walk遍历filePath下的文件、文件夹（包括子目录）
    for parent, dirnames, filenames in os.walk(load_path):
        for filename in filenames:
            image_size = os.path.getsize(os.path.join(parent, filename))
            # 路径、图片尺寸存入字典
            file_map.setdefault(os.path.join(parent, filename), image_size) 

    # 获取的图片列表按image_size排序
    file_map = sorted(file_map.items(), key=lambda d: d[1], reverse=False)
    file_list = []
    for filename, image_size in file_map:
        file_list.append(filename)  

    # 取出重复的图片
    file_repeat = []
    for currIndex, filename in enumerate(file_list):
        dir_image1 = file_list[currIndex]
        dir_image2 = file_list[currIndex + 1]
        result = 比较两张图片是否相同(dir_image1, dir_image2)
        if(result == "两张图片相同"):
            file_repeat.append(file_list[currIndex + 1])
            print("相同的图片：", file_list[currIndex], file_list[currIndex + 1])
        else:
            print("不同的图片：", file_list[currIndex], file_list[currIndex + 1])
        currIndex += 1
        if currIndex >= len(file_list)-1:
            break

    # 将重复的图片移动到新的文件夹
    for image in file_repeat:
        shutil.move(image, save_path)
        print("正在移除重复照片：", image)

    




        
